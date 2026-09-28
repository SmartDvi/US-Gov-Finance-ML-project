"""
MLflow helpers: experiment setup, logging and champion/challenger promotion.

Each pipeline execution creates one parent run ("pipeline") with one nested
run per stage ("forecaster", "clustering", "anomaly_detection"). Every run
gets a `stage` tag so the dashboard can query runs by stage.

Model registry: each forecaster run registers a new model version. The
version is promoted to the "champion" alias only if its test error is lower
than the current champion's; otherwise the old champion stays in place.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import mlflow.spark
import pyspark
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.utils.spark_utils import PROJECT_ROOT

logger = logging.getLogger(__name__)

CHAMPION_ALIAS = "champion"


def tracking_uri(config: dict) -> str:
    """Tracking URI with a relative SQLite path resolved against the project root."""
    uri = config["mlflow"]["tracking_uri"]
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
        uri = prefix + str(PROJECT_ROOT / uri[len(prefix):])
    return uri


def setup_mlflow(config: dict) -> str:
    """Point MLflow at the local database and create the experiment if needed."""
    mlflow.set_tracking_uri(tracking_uri(config))
    name = config["mlflow"]["experiment_name"]
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        artifact_dir = PROJECT_ROOT / config["mlflow"]["artifact_dir"]
        experiment_id = mlflow.create_experiment(name, artifact_location=artifact_dir.as_uri())
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def flatten(d: dict, prefix: str = "", sep: str = "_") -> dict:
    """{'test': {'rf': {'mae': 1}}} -> {'test_rf_mae': 1}"""
    out = {}
    for key, value in d.items():
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, name, sep))
        else:
            out[name] = value
    return out


def log_params(params: dict) -> None:
    mlflow.log_params({k: str(v)[:500] for k, v in flatten(params).items()})


def log_metrics(metrics: dict, prefix: str = "") -> None:
    """Log every numeric value of a (nested) dict as an MLflow metric."""
    numeric = {
        k: float(v) for k, v in flatten(metrics, prefix).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    mlflow.log_metrics(numeric)


def log_spark_model(model, name: str, registered_model_name: str | None = None):
    """Log a fitted Spark ML PipelineModel (and optionally register it)."""
    return mlflow.spark.log_model(
        spark_model=model,
        artifact_path=name,
        registered_model_name=registered_model_name,
        # Listing requirements explicitly skips MLflow's slow auto-detection
        pip_requirements=[f"pyspark=={pyspark.__version__}"],
    )


def promote_if_better(model_name: str, version: str, metric_key: str, new_value: float) -> bool:
    """Move the 'champion' alias to `version` if its metric is lower (better) than the champion's."""
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        champion_value = client.get_run(champion.run_id).data.metrics.get(metric_key)
    except MlflowException:
        champion, champion_value = None, None  # no champion yet

    if champion_value is None or new_value < champion_value:
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
        logger.info("Model %s v%s promoted to champion (%s=%.3f)", model_name, version, metric_key, new_value)
        return True

    logger.info(
        "Model %s v%s kept as challenger (%s=%.3f vs champion v%s %.3f)",
        model_name, version, metric_key, new_value, champion.version, champion_value,
    )
    return False
