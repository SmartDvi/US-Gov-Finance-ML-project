"""Tests for the MLflow helpers, monitoring metrics and the dashboard pages."""

from __future__ import annotations

import mlflow
import pytest
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

from src.features.feature_engineering import build_features
from src.utils.mlflow_utils import CHAMPION_ALIAS, flatten, promote_if_better
from src.utils.monitoring import data_quality_metrics, feature_drift
from tests.conftest import STATES, YEARS


def test_flatten_nested_dict():
    assert flatten({"test": {"rf": {"mae": 1.0}}, "k": 2}) == {"test_rf_mae": 1.0, "k": 2}


def test_data_quality_metrics(clean_df):
    m = data_quality_metrics(clean_df)
    assert m["rows"] == len(STATES) * len(YEARS)
    assert m["states"] == len(STATES)
    assert m["null_cells"] == 0


def test_feature_drift_detects_a_shifted_latest_year(clean_df):
    features = build_features(clean_df)
    shifted = features.withColumn(
        "debt_to_revenue_ratio",
        F.when(F.col("year") == 2019, F.col("debt_to_revenue_ratio") + 10).otherwise(F.col("debt_to_revenue_ratio")),
    )
    drift = feature_drift(shifted, ["debt_to_revenue_ratio", "welfare_share"])
    assert drift["debt_to_revenue_ratio"] > 5          # big, deliberate shift
    assert drift["welfare_share"] < drift["debt_to_revenue_ratio"]


@pytest.fixture
def local_mlflow(tmp_path):
    """Point MLflow at a throw-away SQLite database for one test."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test")
    yield MlflowClient(), tmp_path
    mlflow.set_tracking_uri(None)


def _register_version(client, name: str, mape: float, source: str) -> str:
    with mlflow.start_run() as run:
        mlflow.log_metric("mape", mape)
    return client.create_model_version(name, source=source, run_id=run.info.run_id).version


def test_champion_only_changes_when_a_model_is_better(local_mlflow):
    client, tmp_path = local_mlflow
    name, source = "forecaster", str(tmp_path)
    client.create_registered_model(name)

    v1 = _register_version(client, name, mape=3.0, source=source)
    assert promote_if_better(name, v1, "mape", 3.0)                   # first model becomes champion

    v2 = _register_version(client, name, mape=4.0, source=source)
    assert not promote_if_better(name, v2, "mape", 4.0)               # worse -> stays challenger
    assert client.get_model_version_by_alias(name, CHAMPION_ALIAS).version == v1

    v3 = _register_version(client, name, mape=2.0, source=source)
    assert promote_if_better(name, v3, "mape", 2.0)                   # better -> new champion
    assert client.get_model_version_by_alias(name, CHAMPION_ALIAS).version == v3


def test_dashboard_pages_build():
    """Every page layout builds (shows a 'run the pipeline' alert if there are no outputs yet)."""
    import dash
    from dash._utils import to_json

    import dashboard.app  # noqa: F401  (registers the pages)

    for page in dash.page_registry.values():
        to_json(page["layout"]())
