"""
Data access for the dashboard.

The dashboard only READS what the Spark pipeline produced:
  * parquet tables in output/            (read with pandas: the data is small,
                                          so starting a Spark JVM per page
                                          load would be slow and wasteful)
  * runs and model versions in MLflow    (via the MLflow client)

Nothing is cached, so re-running the pipeline and refreshing the browser
shows the new results.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.utils.mlflow_utils import CHAMPION_ALIAS, tracking_uri
from src.utils.spark_utils import load_config, output_path

CONFIG = load_config()

# Money columns in the source data are in thousands of USD.
THOUSANDS_TO_BILLIONS = 1e-6

# Metrics that can be selected on the dashboard: label -> column in the features table
MONEY_METRICS = {
    "General revenue": "totals_general_revenue",
    "Total revenue": "totals_revenue",
    "Total expenditure": "totals_expenditure",
    "Tax revenue": "totals_tax",
    "Debt at end of year": "totals_debt_at_end_of_fiscal_year",
    "Welfare spending": "details_welfare_welfare_institution_total_expenditure",
    "Education spending": "details_education_education_total",
    "Highway spending": "details_transportation_highways_highways_total_expenditure",
    "Police spending": "details_police_protection",
}


def pretty(name: str) -> str:
    """'debt_to_revenue_ratio' -> 'Debt to revenue ratio'"""
    return name.replace("_", " ").capitalize()


# ── Pipeline outputs ──────────────────────────────────────────────

def _path(*parts: str) -> Path:
    return Path(output_path(CONFIG, *parts))


def pipeline_has_run() -> bool:
    return _path("predictions", "revenue_forecast").exists()


def read_table(*parts: str) -> pd.DataFrame:
    """Read a parquet folder written by Spark."""
    return pd.read_parquet(_path(*parts))


def features() -> pd.DataFrame:
    return read_table("features", "state_finance_features").sort_values(["state", "year"])


def revenue_forecast() -> pd.DataFrame:
    return read_table("predictions", "revenue_forecast")


def forecaster_test_predictions() -> pd.DataFrame:
    return read_table("predictions", "forecaster_test")


def state_clusters() -> pd.DataFrame:
    return read_table("predictions", "state_clusters")


def cluster_profiles() -> pd.DataFrame:
    return read_table("predictions", "cluster_profiles").sort_values("cluster_id")


def anomaly_scores() -> pd.DataFrame:
    return read_table("predictions", "anomaly_scores")


def pipeline_metrics() -> dict:
    with open(_path("reports", "metrics.json")) as fh:
        return json.load(fh)


def state_names() -> list[str]:
    return sorted(features()["state"].unique())


# ── MLflow ────────────────────────────────────────────────────────

def _mlflow_client() -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri(CONFIG))
    return MlflowClient()


def mlflow_runs(stage: str | None = None) -> pd.DataFrame:
    """All runs of the experiment (optionally one stage), oldest first."""
    _mlflow_client()
    runs = mlflow.search_runs(
        experiment_names=[CONFIG["mlflow"]["experiment_name"]],
        filter_string=f"tags.stage = '{stage}'" if stage else "",
        order_by=["attributes.start_time ASC"],
    )
    if runs.empty:
        return runs
    runs["started"] = runs["start_time"].dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M:%S")
    runs["duration_s"] = (runs["end_time"] - runs["start_time"]).dt.total_seconds().round(1)
    return runs


def model_versions() -> pd.DataFrame:
    """Registered forecaster versions with their aliases and test error."""
    client = _mlflow_client()
    name = CONFIG["mlflow"]["registered_model_name"]
    try:
        aliases = client.get_registered_model(name).aliases  # {"champion": "3"}
    except mlflow.exceptions.MlflowException:
        return pd.DataFrame()
    version_to_alias = {str(v): a for a, v in aliases.items()}

    rows = []
    for mv in client.search_model_versions(f"name = '{name}'"):
        run_metrics = client.get_run(mv.run_id).data.metrics
        rows.append({
            "version": int(mv.version),
            "alias": version_to_alias.get(str(mv.version), "challenger"),
            "created": pd.to_datetime(mv.creation_timestamp, unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
            "test_mape_pct": run_metrics.get("test_random_forest_revenue_mape_pct"),
            "baseline_mape_pct": run_metrics.get("test_baseline_average_growth_revenue_mape_pct"),
            "drift_max": run_metrics.get("drift_max"),
            "run_id": mv.run_id,
        })
    return pd.DataFrame(rows).sort_values("version", ascending=False)


def champion_version() -> str | None:
    versions = model_versions()
    champion = versions[versions["alias"] == CHAMPION_ALIAS] if not versions.empty else versions
    return None if champion.empty else str(champion["version"].iloc[0])
