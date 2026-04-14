# Government Finance ML Pipeline
### Production-Grade PySpark + MLflow + Databricks

> **Mission:** Transform raw US state government financial data (1992–2019) into actionable policy intelligence using a four-model ML pipeline — fiscal forecasting, anomaly detection, state clustering, and resource optimization.

---

## Architecture Overview

```
Raw CSV
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  BRONZE LAYER  (Delta Lake)                                     │
│  • Schema enforcement & sanitisation                            │
│  • Auto Loader streaming support                                │
│  • Audit columns (_ingested_at, _row_id)                        │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  SILVER LAYER  (Delta Lake)                                     │
│  • Deduplication  •  Year-range filter                         │
│  • IQR outlier clipping  •  Rolling median imputation          │
│  • Data Quality Score (DQS)  •  Train/test split               │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  GOLD LAYER  (Delta Lake)                                       │
│  • 60+ features: ratios, YoY growth, lags, rolling stats       │
│  • Fiscal Health Index  •  Social Investment Score             │
│  • Debt sustainability flags                                    │
└──────┬──────────────┬──────────────┬──────────────┬────────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
 ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐
 │ MODEL A  │  │ MODEL B  │  │ MODEL C  │  │   MODEL D    │
 │ Fiscal   │  │ Anomaly  │  │ State    │  │  Resource    │
 │ Forecast │  │ Detector │  │Clustering│  │  Optimizer   │
 │          │  │          │  │          │  │              │
 │ Prophet  │  │ Isolation│  │ K-Means  │  │  Genetic     │
 │ XGBoost  │  │ Forest + │  │ (PySpark │  │  Algorithm   │
 │ Pandas   │  │ Autoenc. │  │   ML)    │  │  (Pareto)    │
 │   UDF    │  │ Ensemble │  │          │  │              │
 └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘
      │              │              │               │
      └──────────────┴──────────────┴───────────────┘
                              │
                              ▼
                    MLflow Model Registry
                    (Unity Catalog — @champion)
```

---

## Business Insights Addressed

| Insight | Model | Output |
|---------|-------|--------|
| Social safety-net efficiency | Forecaster + Feature Ratios | `welfare_to_insurance_ratio`, FHI trend |
| Infrastructure vs. mobility | Optimizer | Optimal highway spend share per state |
| Public safety vs. community investment | Clustering + Optimizer | `police_vs_parks_ratio` cluster profiles |
| Fiscal health & future taxation | Forecaster + Anomaly | Debt trajectory + early-warning flags |

---

## Project Structure

```
gov_finance_ml/
├── config/
│   ├── config.yaml              # Master configuration (all tunable params)
│   └── databricks_job.json      # Databricks multi-task job definition
│
├── src/
│   ├── data/
│   │   ├── ingestion.py         # Bronze ingestion (CSV, Auto Loader)
│   │   └── preprocessing.py     # Silver: clean, impute, DQS, split
│   ├── features/
│   │   └── feature_engineering.py  # Gold: 60+ features + ML pipeline
│   ├── models/
│   │   ├── fiscal_forecaster.py    # Prophet + XGBoost (Model A)
│   │   ├── anomaly_detector.py     # IF + Autoencoder (Model B)
│   │   ├── state_clustering.py     # KMeans + profiling (Model C)
│   │   └── resource_optimizer.py   # Genetic Algorithm (Model D)
│   └── utils/
│       ├── spark_utils.py       # Session factory, Delta helpers
│       └── mlflow_utils.py      # Experiment, run, registry helpers
│
├── notebooks/
│   ├── 01_data_ingestion.py     # Databricks notebook: Bronze + Silver
│   ├── 02_feature_engineering.py
│   ├── 03_fiscal_forecasting.py
│   ├── 04_anomaly_detection.py
│   ├── 05_state_clustering.py
│   └── 06_resource_optimization.py
│
├── tests/
│   └── test_pipeline.py         # Unit + integration tests (pytest)
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quickstart on Databricks

### 1. Clone the Repo
```bash
# In Databricks Workspace → Repos → Add Repo
# URL: https://github.com/yourorg/gov_finance_ml
```

### 2. Upload Data
```bash
# Upload your CSV to DBFS:
dbfs cp state_finances_raw.csv dbfs:/FileStore/gov_finance/state_finances_raw.csv
```

### 3. Create Unity Catalog (one-time)
Run the SQL block in `01_data_ingestion.py`:
```sql
CREATE CATALOG IF NOT EXISTS gov_finance;
CREATE SCHEMA IF NOT EXISTS gov_finance.bronze;
CREATE SCHEMA IF NOT EXISTS gov_finance.silver;
CREATE SCHEMA IF NOT EXISTS gov_finance.gold;
```

### 4. Run Notebooks in Order
| Notebook | Runtime |
|----------|---------|
| `01_data_ingestion` | ~5 min |
| `02_feature_engineering` | ~8 min |
| `03_fiscal_forecasting` | ~25 min |
| `04_anomaly_detection` | ~10 min |
| `05_state_clustering` | ~12 min |
| `06_resource_optimization` | ~30 min |

### 5. Schedule with Job API
```bash
databricks jobs create --json @config/databricks_job.json
```

---

## MLflow Experiment Structure

```
/Shared/gov_finance_ml/
├── fiscal_forecasting/          # Prophet & XGBoost runs per target
├── anomaly_detection/           # Ensemble IF + AE runs
├── state_clustering/            # KMeans K-selection runs
└── resource_optimization/       # GA optimizer per state
```

**UC Model Registry aliases:**
- `gov_finance.fiscal_forecaster_totals_revenue@champion`
- `gov_finance.fiscal_forecaster_totals_expenditure@champion`
- `gov_finance.anomaly_detector@champion`
- `gov_finance.state_clustering@champion`

---

## Configuration

All tuneable parameters are in `config/config.yaml`:

```yaml
models:
  fiscal_forecaster:
    horizon_years: 5           # Forecast horizon
    xgb:
      n_estimators: [100, 300, 500]   # Grid search values

  anomaly_detector:
    isolation_forest:
      contamination: 0.05      # Expected anomaly fraction

  clustering:
    k_range: [3, 4, 5, 6, 7, 8]   # K candidates for elbow / silhouette

  optimizer:
    objective_weights:
      minimize_debt: 0.35
      maximize_welfare: 0.30
      maximize_infrastructure: 0.20
      maximize_public_safety: 0.15
```

---

## Running Tests Locally

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run linter
ruff check src/ tests/
```

---

## Output Tables Summary

| Table | Content |
|-------|---------|
| `bronze.state_finances_raw` | Raw ingested data |
| `silver.state_finances` | Cleaned, deduplicated, imputed |
| `gold.state_finances_features` | 60+ ML features |
| `gold.state_finances_features_anomaly_scores` | Per-record anomaly flags |
| `gold.state_finances_features_cluster_labels` | State cluster assignments |
| `gold.state_finances_features_optimal_allocations` | GA budget recommendations |

---

## Key Features by Model

### Model A — Fiscal Forecaster
- Distributed per-state training via PySpark Pandas UDF
- Prophet handles structural breaks (policy changes, recessions)
- XGBoost captures non-linear lag relationships
- MLflow grid search with automatic champion registration

### Model B — Anomaly Detector
- Isolation Forest: no distributional assumptions, fast at scale
- Autoencoder: learns "normal state fiscal DNA", flags reconstruction errors
- Ensemble score combines both for higher precision
- Severity tiers: `critical → high → normal → clean`

### Model C — State Clustering
- Native PySpark ML KMeans (distributed, no data collection to driver)
- Silhouette + inertia elbow for optimal K
- Human-readable cluster archetypes (post-hoc labelling)
- Peer-state lookup API for benchmarking

### Model D — Resource Optimizer
- Genetic Algorithm with Pareto-optimal multi-objective fitness
- Configurable objective weights (welfare vs. infrastructure vs. debt)
- Hard constraint enforcement per budget category
- Sensitivity analysis: how does optimal allocation shift with priority weights?

---

