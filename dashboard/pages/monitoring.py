"""Page 5 — Model monitoring: MLflow runs, metric trends, feature drift and the model registry."""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
from dash import Input, Output, callback

from dashboard import data
from dashboard.components import (
    DECIMAL_2, data_grid, kpi_card, kpi_row, page_header, section, to_records,
)

dash.register_page(
    __name__, path="/monitoring", name="Model Monitoring",
    description="MLflow runs, drift & registry", order=4,
)

DRIFT_THRESHOLD = data.CONFIG["monitoring"]["drift_threshold"]
MLFLOW_UI_COMMAND = f"uv run mlflow ui --backend-store-uri {data.CONFIG['mlflow']['tracking_uri']}"

# Metrics that can be trended across runs: (stage tag, metric key, label)
TREND_METRICS = {
    "Forecaster": [
        ("forecaster", "test_random_forest_revenue_mape_pct", "Test MAPE % — random forest"),
        ("forecaster", "test_baseline_average_growth_revenue_mape_pct", "Test MAPE % — average-growth baseline"),
        ("forecaster", "drift_max", "Max feature drift (std)"),
        ("forecaster", "forecast_mean_predicted_growth", "Mean predicted growth"),
    ],
    "Clustering": [("clustering", "silhouette", "Silhouette score")],
    "Anomaly detection": [
        ("anomaly_detection", "anomaly_rate_pct", "Anomaly rate %"),
        ("anomaly_detection", "anomalies_flagged", "Anomalies flagged"),
    ],
    "Data quality": [
        ("pipeline", "data_rows", "Rows ingested"),
        ("pipeline", "data_null_cells", "Null cells"),
    ],
}

REGISTRY_COLUMNS = [
    {"field": "version", "headerName": "Version", "maxWidth": 110},
    {"field": "alias", "headerName": "Alias",
     "cellStyle": {"styleConditions": [{"condition": "params.value == 'champion'",
                                        "style": {"color": "var(--mantine-color-teal-7)", "fontWeight": 700}}]}},
    {"field": "created", "headerName": "Registered"},
    {"field": "test_mape_pct", "headerName": "Test MAPE %", "valueFormatter": DECIMAL_2},
    {"field": "baseline_mape_pct", "headerName": "Baseline MAPE %", "valueFormatter": DECIMAL_2},
    {"field": "drift_max", "headerName": "Max drift", "valueFormatter": DECIMAL_2},
    {"field": "run_id", "headerName": "MLflow run id", "minWidth": 260},
]

RUN_COLUMNS = [
    {"field": "started", "headerName": "Started", "sort": "desc", "minWidth": 170},
    {"field": "tags.stage", "headerName": "Stage"},
    {"field": "status", "headerName": "Status"},
    {"field": "duration_s", "headerName": "Duration (s)"},
    {"field": "metrics.test_random_forest_revenue_mape_pct", "headerName": "Test MAPE %", "valueFormatter": DECIMAL_2},
    {"field": "metrics.drift_max", "headerName": "Max drift", "valueFormatter": DECIMAL_2},
    {"field": "metrics.silhouette", "headerName": "Silhouette", "valueFormatter": DECIMAL_2},
    {"field": "metrics.anomaly_rate_pct", "headerName": "Anomaly %", "valueFormatter": DECIMAL_2},
    {"field": "run_id", "headerName": "Run id", "minWidth": 260},
]


def _no_runs() -> dmc.Alert:
    return dmc.Alert("No MLflow runs yet. Run `uv run python main.py` to create one.",
                     title="Nothing tracked yet", color="yellow")


def _drift_chart(forecaster_runs) -> dmc.BarChart:
    """Per-feature drift of the latest run, split into within / over threshold (two colours)."""
    latest = forecaster_runs.iloc[-1]
    drift = {c.removeprefix("metrics.drift_"): latest[c] for c in forecaster_runs.columns
             if c.startswith("metrics.drift_") and c not in ("metrics.drift_max", "metrics.drift_features_over_threshold")}
    rows = [
        {"feature": data.pretty(f),
         "within": v if v <= DRIFT_THRESHOLD else None,
         "over": v if v > DRIFT_THRESHOLD else None}
        for f, v in sorted(drift.items(), key=lambda kv: -kv[1])
    ]
    return dmc.BarChart(
        h=420,
        dataKey="feature",
        data=rows,
        type="stacked",
        orientation="vertical",
        series=[{"name": "within", "label": "Within threshold", "color": "blue.6"},
                {"name": "over", "label": "Over threshold", "color": "red.6"}],
        referenceLines=[{"x": DRIFT_THRESHOLD, "label": f"threshold {DRIFT_THRESHOLD}", "color": "red.6"}],
        valueFormatter={"function": "decimal2"},
        yAxisProps={"width": 230},
        withLegend=True,
    )


def layout(**_):
    pipeline_runs = data.mlflow_runs("pipeline")
    if pipeline_runs.empty:
        return _no_runs()
    forecaster_runs = data.mlflow_runs("forecaster")
    latest_forecaster = forecaster_runs.iloc[-1]
    versions = data.model_versions()
    champion = versions[versions["alias"] == "champion"].iloc[0] if not versions.empty else None
    drift_max = latest_forecaster["metrics.drift_max"]
    over = int(latest_forecaster["metrics.drift_features_over_threshold"])

    return dmc.Container(
        [
            page_header(
                "Model Monitoring",
                "Every pipeline run is tracked in MLflow: parameters, metrics, drift checks and models. "
                "A new forecaster only becomes 'champion' if its test error beats the current champion.",
            ),
            kpi_row([
                kpi_card("Pipeline runs", str(len(pipeline_runs)), f"last: {pipeline_runs['started'].iloc[-1]}"),
                kpi_card("Last run status", pipeline_runs["status"].iloc[-1],
                         color="teal" if pipeline_runs["status"].iloc[-1] == "FINISHED" else "red"),
                kpi_card("Champion model", f"v{champion['version']}" if champion is not None else "–",
                         f"test MAPE {champion['test_mape_pct']:.2f}%" if champion is not None else None,
                         color="indigo"),
                kpi_card("Max feature drift", f"{drift_max:.2f} std",
                         f"{over} feature(s) over {DRIFT_THRESHOLD}",
                         color="red" if drift_max > DRIFT_THRESHOLD else "teal"),
            ]),
            section(
                "Metric trend across runs",
                dmc.Stack([
                    dmc.Select(
                        id="monitor-metric",
                        label="Metric",
                        data=[{"group": group, "items": [{"value": f"{stage}|{key}", "label": label}
                                                         for stage, key, label in items]}
                              for group, items in TREND_METRICS.items()],
                        value="forecaster|test_random_forest_revenue_mape_pct",
                        allowDeselect=False,
                        w=380,
                    ),
                    dmc.LineChart(id="monitor-trend", h=280, dataKey="run", data=[],
                                  series=[{"name": "value", "label": "Value", "color": "indigo.6"}],
                                  withDots=True, valueFormatter={"function": "decimal2"}),
                ]),
                description="One point per run, so you can spot degradation after new data arrives.",
            ),
            section(
                "Feature drift — latest year vs history (latest run)",
                _drift_chart(forecaster_runs),
                description="Drift = |mean(latest year) − mean(earlier years)| / std(earlier years). "
                            "Large drift means the model is being asked about conditions it has rarely seen.",
            ),
            section("Model registry — " + data.CONFIG["mlflow"]["registered_model_name"],
                    data_grid("monitor-registry", versions, REGISTRY_COLUMNS, height=260)),
            section("Run history", data_grid("monitor-runs", data.mlflow_runs(), RUN_COLUMNS, height=360)),
            section(
                "Open the full MLflow UI",
                dmc.Stack([
                    dmc.Code(MLFLOW_UI_COMMAND, block=True),
                    dmc.Anchor("http://127.0.0.1:5000", href="http://127.0.0.1:5000", target="_blank"),
                ], gap="xs"),
            ),
        ],
        fluid=True,
    )


@callback(Output("monitor-trend", "data"), Input("monitor-metric", "value"))
def update_trend(selection):
    stage, key = selection.split("|")
    runs = data.mlflow_runs(stage)
    column = f"metrics.{key}"
    if runs.empty or column not in runs:
        return []
    return to_records(runs[["started", column]].set_axis(["run", "value"], axis=1))
