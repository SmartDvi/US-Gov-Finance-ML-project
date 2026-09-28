# US State Government Finance — PySpark ML Pipeline

An end-to-end **PySpark** pipeline that turns 21 years of US state government
finance data into three answers that budget analysts and auditors need:

| Question | Model | Output |
|----------|-------|--------|
| How much general revenue will each state collect next year? | **Revenue forecaster** — Spark ML `RandomForestRegressor` | `reports/revenue_forecast` |
| Which states are true peers, and where does a state spend differently from them? | **Peer clustering** — Spark ML `KMeans` | `reports/state_clusters`, `reports/cluster_profiles` |
| Which state-years had unusual swings that deserve an audit? | **Anomaly detection** — robust z-score (median / MAD) in Spark SQL | `reports/anomaly_audit` |

Every run is tracked and monitored in **MLflow** (metrics, drift, model registry with
champion/challenger promotion). Results are explored in a multi-page **Dash**
dashboard built with Dash Mantine Components and Dash AG Grid.

---

## Data

`data/raw/finance.csv`: US Census *Annual Survey of State Government Finances*
(via CORGIS). One row per state per year, about 30 columns in **thousands of USD**
(revenue, expenditure, debt, tax, welfare, education, highways, police, ...).

Facts about the data that shaped the design:

| Fact | Consequence |
|------|-------------|
| Years **1992–2004 and 2012–2019** (2005–2011 missing) | Growth rates and lags are only computed between consecutive years (`lag_if_consecutive`). A plain `lag()` would call 2004→2012 a one-year change. |
| Contains a `UNITED STATES` aggregate row | Removed during ingestion; otherwise it would dominate every model. |
| California's budget is ~100× Wyoming's | Models use **ratios and growth rates**, not raw dollars. |
| Insurance-trust revenue is sometimes negative (pension investment losses in 2001–2003) | Kept as real data, not clipped. Also the reason the forecaster targets *general* revenue. |

---

## Pipeline

```text
data/raw/finance.csv
        │  Stage 1  src/data/ingestion.py
        ▼  snake_case columns, typed, "UNITED STATES" removed, validated
output/clean/state_finances            (parquet)
        │  Stage 2  src/features/feature_engineering.py
        ▼  spending shares, revenue mix, fiscal-stress ratios, YoY growth
output/features/state_finance_features (parquet, 1 row per state-year)
        │
        ├── Stage 3a  src/models/fiscal_forecaster.py  → revenue forecast for next year
        ├── Stage 3b  src/models/state_clustering.py   → peer groups + gaps vs peers
        └── Stage 3c  src/models/anomaly_detector.py   → audit list with reasons
                                                          │
                                         output/reports/*.csv, metrics.json
```

### Stage 2 — features (all computed with Spark window functions)

* **Spending shares**: welfare, education, health, highways, police, corrections and parks, each as a share of general expenditure.
* **Revenue mix**: tax share, intergovernmental (federal aid) dependency, insurance-trust share.
* **Fiscal stress**: expenditure / revenue, debt / general revenue, interest burden, deficit flag.
* **Growth**: year-over-year growth of revenue, tax, expenditure, debt, capital outlay, interest and miscellaneous revenue, plus last year's growth and the national average growth for that year.
* Every feature uses only the current or earlier years, so there is no look-ahead.

### Stage 3a — revenue forecaster

* **Label** = next year's general-revenue *growth rate*. Forecast in dollars = this year's revenue × (1 + predicted growth).
  Growth rates are used because tree models can't predict values above the training range, and revenue grows every year. Growth rates are also comparable across states.
* **Time-based split** (never random for time series): train on target years < 2015, pick hyper-parameters on 2015–2016, test once on 2017–2019.
* The final model is refit on all years and forecasts **2020** for every state.
* It is always compared with two baselines: *naive* (no change) and *historical average growth*.

### Stage 3b — peer clustering

* One profile per state: the average of 10 budget-structure ratios over the last 5 years.
* `StandardScaler` → `KMeans` for k = 3…6, choosing the k with the best **silhouette** score.
* Each cluster is described automatically by its three most distinctive features.
* **Peer benchmark**: every ratio minus the cluster median, for example `welfare_share_vs_peers`.

### Stage 3c — anomaly detection

* Features: year-over-year growth of revenue, expenditure, capital outlay, debt, interest and miscellaneous revenue.
* Modified z-score `0.6745 × (x − median) / MAD`. The median and MAD are used because the outliers we're hunting would inflate a mean and standard deviation and hide themselves.
* A row is flagged when its largest |z| is above 5.0. The `reasons` column names each metric that triggered the flag.

---

## Results (from `output/reports/metrics.json`)

**Revenue forecaster.** Test years 2017–2019, 150 state-years never seen in training:

| Model | Growth MAE (pp) | Revenue MAPE |
|-------|----------------:|-------------:|
| Random forest | 2.83 | 2.71 % |
| Baseline: no change | 4.68 | 4.40 % |
| Baseline: historical average growth | 2.60 | 2.48 % |

On the validation years (2015–2016) the forest beat the average-growth baseline (3.56 vs 3.98 pp).
On the calmer 2017–2019 test years it is on par with it. So it forecasts next year's revenue within about 2.7 %,
far better than assuming no change, but the historical trend already explains most of the signal.
The most important features are national revenue growth, federal-aid dependency and tax share.

**Peer clustering.** k = 4, silhouette 0.22. For example:

* **California** spends 41.8 % of its general budget on welfare, **+11.1 pp more than its peer group's median**, and 4.7 pp less on education.
* **Texas** spends 4.3 pp more than its peers on education, 2.4 pp more on highways, and about the same on welfare.

**Anomaly detection.** 55 of 950 state-years flagged (5.8 %). The top of the list includes West Virginia 2018
(interest on general debt jumped, z = +21.5) and Kansas 2000–2004 (repeated debt and interest spikes).

---

## MLflow tracking and monitoring

Each `main.py` run creates one parent MLflow run (`pipeline`) and a nested run per stage.
Everything goes into a local SQLite database (`mlflow.db`) with artifacts in `mlartifacts/`.

| Run (tag `stage`) | Params | Metrics | Artifacts |
|---|---|---|---|
| `pipeline` | config sections | data quality: rows, states, years, null cells | `metrics.json` |
| `forecaster` | target, best depth / trees | test MAE/RMSE/MAPE for the model **and** both baselines; drift per feature; prediction summary | Spark model, feature importance, tuning results, forecast CSV |
| `clustering` | k range, selected k | silhouette (plus a silhouette-by-k curve) | Spark model, cluster reports |
| `anomaly_detection` | threshold, features | rows scored, anomalies, anomaly rate | fitted medians/MADs, audit list |

**Monitoring on every run** (`src/utils/monitoring.py`):

* **Data quality:** did the expected rows, states and years arrive, and how many nulls?
* **Feature drift:** `|mean(latest year) − mean(earlier years)| / std(earlier years)` for each forecaster feature. Above 0.5 std counts as drift. The model is then being asked about conditions it has rarely seen.
* **Prediction sanity:** mean, min and max predicted growth.

**Model registry, champion/challenger:** each run registers a new version of
`state_revenue_forecaster`. `promote_if_better()` moves the `champion` alias only when
the new version's test MAPE is *lower* than the current champion's. Otherwise it stays a challenger.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```

## Dashboard

```bash
uv run python -m dashboard.app                               # http://127.0.0.1:8050
```

A multi-page app (`dash.register_page`) that reads the pipeline's Parquet outputs with pandas
and MLflow with the MLflow client. It doesn't start Spark: the heavy work is already done.

| Page | What you can do | Components |
|---|---|---|
| **Overview** | Headline KPIs; compare any metric over time for chosen states | `Select` (metric), `MultiSelect` (states), `LineChart`, AG Grid |
| **Revenue Forecast** | Model vs baselines; next-year forecast per state; one state's history plus forecast | `MultiSelect`, `Select` (weakest/strongest), `BarChart`, AG Grid |
| **Peer Clusters** | Cluster cards; peer benchmark table; how a state differs from its peers | `Select` (cluster, state), `MultiSelect` (features), AG Grid with coloured gaps |
| **Anomaly Audit** | Filter the audit list by state, metric and years | `MultiSelect` ×2, `Select`, `RangeSlider`, `BarChart`, AG Grid |
| **Model Monitoring** | Run history, metric trend across runs, drift chart with threshold, model registry | grouped `Select`, `LineChart`, `BarChart`, AG Grid |

Pages rebuild on every visit, so re-running the pipeline and refreshing the browser shows the new results.

## How to run

Requirements: Python 3.12, Java 17 or 21, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # install dependencies
uv run python main.py                    # pipeline + MLflow tracking (~2 minutes)
uv run python -m dashboard.app           # dashboard  -> http://127.0.0.1:8050
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # MLflow UI -> http://127.0.0.1:5000
uv run pytest                            # 23 tests (~2 minutes)
```

Always use the project's own environment (`uv run ...`, or `source .venv/bin/activate` first).
A conda `(base)` Python doesn't have the right packages (e.g. `dash_ag_grid`) and will fail to import them.

All settings (paths, Spark memory, train/validation/test years, hyper-parameter
grid, k range, anomaly threshold, MLflow, drift threshold, dashboard port) are in
`config/config.yaml`, with a comment explaining each choice.

## Project structure

```text
config/config.yaml                  all tunable settings
data/raw/finance.csv                source data
main.py                             runs the pipeline, logs everything to MLflow
src/data/ingestion.py               Stage 1: load, clean, validate
src/features/feature_engineering.py Stage 2: ratios and growth features
src/models/fiscal_forecaster.py     Stage 3a: RandomForest revenue forecaster
src/models/state_clustering.py      Stage 3b: KMeans peer groups + benchmark
src/models/anomaly_detector.py      Stage 3c: robust z-score anomalies
src/utils/spark_utils.py            config, SparkSession, parquet/CSV I/O
src/utils/mlflow_utils.py           MLflow setup, logging, champion/challenger promotion
src/utils/monitoring.py             data quality, feature drift, prediction summary
dashboard/app.py                    Dash app shell (header, navigation, page container)
dashboard/data.py                   reads pipeline outputs and MLflow for the pages
dashboard/components.py             shared cards, sections, AG Grid defaults, formatters
dashboard/pages/                    overview, forecast, clusters, anomalies, monitoring
dashboard/assets/dmc_functions.js   chart value formatters (functions-as-props)
tests/                              pytest tests (synthetic data)
output/, mlflow.db, mlartifacts/    generated — not committed
```

## Limitations and next steps

* 21 years × 50 states is a small dataset. Adding economic drivers such as state GDP, unemployment and population would help the forecaster more than a more complex model would.
* Anomalies are *statistically unusual*, not proof of fraud. Some are real policy events (debt refinancing, new bond programmes), which is why each flag comes with its reason.
* Dollar amounts are nominal. Adjusting for inflation would make long-run comparisons fairer.
