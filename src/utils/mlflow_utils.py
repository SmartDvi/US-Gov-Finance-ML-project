"""
gov_finance_ml/src/utils/mlflow_utils.py
Production-grade MLflow helpers for Databricks Unity Catalog.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Optional

import mlflow
import mlflow.sklearn
import mlflow.spark
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Experiment management
# ─────────────────────────────────────────────

def setup_mlflow(
    experiment_path: str,
    tracking_uri: str = "databricks",
    registry_uri: str = "databricks-uc",
) -> str:
    """
    Configure MLflow for Databricks Unity Catalog.
    Returns the experiment ID.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)

    client = MlflowClient()
    try:
        exp = client.get_experiment_by_name(experiment_path)
        if exp is None:
            exp_id = client.create_experiment(experiment_path)
            logger.info("Created MLflow experiment: %s  (id=%s)", experiment_path, exp_id)
        else:
            exp_id = exp.experiment_id
            logger.info("Using existing MLflow experiment: %s  (id=%s)", experiment_path, exp_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not reach MLflow server (%s). Running without tracking.", exc)
        exp_id = "local"

    mlflow.set_experiment(experiment_path)
    return exp_id


# ─────────────────────────────────────────────
# Run context manager
# ─────────────────────────────────────────────

@contextmanager
def managed_run(
    run_name: str,
    tags: Optional[dict[str, str]] = None,
    nested: bool = False,
):
    """
    Context manager that starts / ends an MLflow run and handles exceptions.

    Usage::
        with managed_run("train_xgb", tags={"model": "xgb"}) as run:
            mlflow.log_param("lr", 0.01)
            mlflow.log_metric("rmse", 1.23)
    """
    _tags = {
        "project": "gov_finance_ml",
        "env": os.getenv("ENV", "development"),
    }
    if tags:
        _tags.update(tags)

    with mlflow.start_run(run_name=run_name, tags=_tags, nested=nested) as run:
        start = time.time()
        logger.info("MLflow run started → %s  (run_id=%s)", run_name, run.info.run_id)
        try:
            yield run
        except Exception:
            mlflow.set_tag("run_status", "FAILED")
            raise
        finally:
            elapsed = round(time.time() - start, 2)
            mlflow.log_metric("run_duration_seconds", elapsed)
            logger.info(
                "MLflow run finished → %s  (%.1fs)  run_id=%s",
                run_name, elapsed, run.info.run_id,
            )


# ─────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────

def log_params_flat(params: dict[str, Any], prefix: str = "") -> None:
    """Log a (potentially nested) dict as flat MLflow params."""
    flat = _flatten(params, prefix)
    # MLflow caps param values at 500 chars
    flat = {k: str(v)[:500] for k, v in flat.items()}
    mlflow.log_params(flat)


def log_metrics_dict(metrics: dict[str, float], step: Optional[int] = None) -> None:
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v), step=step)


def log_json_artifact(obj: Any, filename: str) -> None:
    """Serialize obj to JSON and log as an MLflow artifact."""
    import tempfile, pathlib
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump(obj, fh, indent=2, default=str)
        tmp_path = fh.name
    mlflow.log_artifact(tmp_path, artifact_path="artifacts")
    pathlib.Path(tmp_path).unlink(missing_ok=True)


def _flatten(d: dict, prefix: str = "", sep: str = ".") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        else:
            out[key] = v
    return out


# ─────────────────────────────────────────────
# Model registration (Unity Catalog)
# ─────────────────────────────────────────────

def register_model(
    run_id: str,
    artifact_path: str,
    registered_name: str,
    alias: str = "challenger",
    description: str = "",
) -> str:
    """
    Register a model to UC Model Registry and assign an alias.
    Returns the model version string.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    client = MlflowClient()

    mv = mlflow.register_model(model_uri=model_uri, name=registered_name)
    version = mv.version

    if description:
        client.update_model_version(
            name=registered_name, version=version, description=description
        )

    client.set_registered_model_alias(
        name=registered_name, alias=alias, version=version
    )
    logger.info(
        "Registered model '%s' v%s with alias '%s'", registered_name, version, alias
    )
    return version


def promote_alias(
    registered_name: str,
    version: str,
    from_alias: str = "challenger",
    to_alias: str = "champion",
) -> None:
    """Atomically move a version from challenger → champion."""
    client = MlflowClient()
    client.set_registered_model_alias(
        name=registered_name, alias=to_alias, version=version
    )
    logger.info(
        "Promoted '%s' v%s: %s → %s", registered_name, version, from_alias, to_alias
    )


def load_champion(registered_name: str) -> Any:
    """Load the current champion model by alias."""
    uri = f"models:/{registered_name}@champion"
    logger.info("Loading champion model from %s", uri)
    return mlflow.sklearn.load_model(uri)


# ─────────────────────────────────────────────
# Signature inference
# ─────────────────────────────────────────────

def build_signature(X, y_pred):
    """Infer MLflow model signature from feature array and predictions."""
    return infer_signature(X, y_pred)
