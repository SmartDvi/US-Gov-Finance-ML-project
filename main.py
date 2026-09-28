"""
Run the full pipeline and track it in MLflow:

    raw CSV -> clean -> features -> forecaster + clustering + anomaly detection

Usage:
    uv run python main.py                      # uses config/config.yaml
    uv run python main.py --config my.yaml

Outputs (paths from config.yaml):
    output/clean/, output/features/     parquet tables
    output/predictions/                 parquet tables read by the dashboard
    output/reports/*.csv, metrics.json  results for humans
    mlflow.db + mlartifacts/            MLflow runs, metrics, models, registry
"""

from __future__ import annotations

import argparse
import logging

import mlflow
from pyspark.sql import DataFrame

from src.data.ingestion import run_ingestion
from src.features.feature_engineering import build_features
from src.models.anomaly_detector import AnomalyDetector, audit_report
from src.models.fiscal_forecaster import FEATURE_COLS, FiscalForecaster
from src.models.state_clustering import StateClusteringModel
from src.utils.mlflow_utils import (
    log_metrics, log_params, log_spark_model, promote_if_better, setup_mlflow,
)
from src.utils.monitoring import data_quality_metrics, feature_drift, prediction_summary
from src.utils.spark_utils import (
    PROJECT_ROOT, get_spark, load_config, output_path, read_parquet,
    write_csv_report, write_json, write_parquet,
)

logger = logging.getLogger("pipeline")

# Metric used to decide whether a new forecaster replaces the champion
PROMOTION_METRIC = "test_random_forest_revenue_mape_pct"


def main(config_path: str | None = None) -> dict:
    config = load_config(config_path)
    spark = get_spark(config)
    setup_mlflow(config)

    with mlflow.start_run(run_name="pipeline", tags={"stage": "pipeline"}) as run:
        log_params({k: config[k] for k in ("data", "forecaster", "clustering", "anomaly")})

        # ── Stage 1: ingestion ──────────────────────────────────
        clean_df = run_ingestion(
            spark,
            csv_path=str(PROJECT_ROOT / config["paths"]["raw_csv"]),
            exclude_states=config["data"]["exclude_states"],
        )
        write_parquet(clean_df, output_path(config, "clean", "state_finances"))
        log_metrics(data_quality_metrics(clean_df), prefix="data")

        # ── Stage 2: features ───────────────────────────────────
        features_path = output_path(config, "features", "state_finance_features")
        write_parquet(build_features(clean_df), features_path)
        # Read back so the models start from the saved table, not a long lineage
        features_df = read_parquet(spark, features_path).cache()

        # ── Stage 3: models (one nested MLflow run each) ────────
        forecast = run_forecaster(config, features_df)
        clusters = run_clustering(config, features_df)
        anomalies = run_anomaly_detection(config, features_df)

        metrics = {
            "forecaster": forecast["metrics"],
            "clustering": clusters["metrics"],
            "anomaly_detection": anomalies["metrics"],
        }
        metrics_path = output_path(config, "reports", "metrics.json")
        write_json(metrics, metrics_path)
        mlflow.log_artifact(metrics_path)
        logger.info("MLflow pipeline run: %s", run.info.run_id)

    _print_summary(forecast, clusters, anomalies, metrics)
    spark.stop()
    return metrics


# ── Stages ────────────────────────────────────────────────────────

def run_forecaster(config: dict, features_df: DataFrame) -> dict:
    with mlflow.start_run(run_name="forecaster", nested=True, tags={"stage": "forecaster"}):
        result = FiscalForecaster(config).run(features_df)
        m = result["metrics"]

        log_params({"target": m["target"], "n_features": len(FEATURE_COLS), **m["best_params"]})
        log_metrics(m["test"], prefix="test")
        log_metrics(prediction_summary(result["forecast"]), prefix="forecast")
        mlflow.log_dict(m["feature_importance"], "feature_importance.json")
        mlflow.log_dict({"tuning_results": m["tuning_results"]}, "tuning_results.json")

        # Monitoring: has the latest year drifted away from history?
        drift = feature_drift(features_df, FEATURE_COLS)
        threshold = config["monitoring"]["drift_threshold"]
        log_metrics(drift, prefix="drift")
        mlflow.log_metrics({
            "drift_max": max(drift.values()),
            "drift_features_over_threshold": sum(v > threshold for v in drift.values()),
        })
        m["drift"] = drift

        # Save, register and (maybe) promote the model
        model_name = config["mlflow"]["registered_model_name"]
        model_info = log_spark_model(result["model"], name="model", registered_model_name=model_name)
        m["model_version"] = model_info.registered_model_version
        m["promoted_to_champion"] = promote_if_better(
            model_name, m["model_version"], PROMOTION_METRIC,
            m["test"]["random_forest"]["revenue_mape_pct"],
        )
        mlflow.set_tag("promoted_to_champion", m["promoted_to_champion"])

        write_parquet(result["test_predictions"], output_path(config, "predictions", "forecaster_test"))
        write_parquet(result["forecast"], output_path(config, "predictions", "revenue_forecast"))
        report_path = output_path(config, "reports", "revenue_forecast")
        write_csv_report(result["forecast"].orderBy("predicted_growth"), report_path)
        mlflow.log_artifacts(report_path, "reports/revenue_forecast")
    return result


def run_clustering(config: dict, features_df: DataFrame) -> dict:
    with mlflow.start_run(run_name="clustering", nested=True, tags={"stage": "clustering"}):
        result = StateClusteringModel(config).run(features_df)
        m = result["metrics"]

        log_params({"k_range": config["clustering"]["k_range"], "profile_years": m["profile_years"],
                    "selected_k": m["selected_k"]})
        mlflow.log_metric("silhouette", m["selected_silhouette"])
        for k, score in m["silhouette_by_k"].items():
            mlflow.log_metric("silhouette_by_k", score, step=int(k))  # draws a curve in the UI
        log_spark_model(result["model"], name="model")

        write_parquet(result["state_clusters"], output_path(config, "predictions", "state_clusters"))
        write_parquet(result["cluster_profiles"], output_path(config, "predictions", "cluster_profiles"))
        for report in ("state_clusters", "cluster_profiles"):
            path = output_path(config, "reports", report)
            write_csv_report(result[report], path)
            mlflow.log_artifacts(path, f"reports/{report}")
    return result


def run_anomaly_detection(config: dict, features_df: DataFrame) -> dict:
    with mlflow.start_run(run_name="anomaly_detection", nested=True, tags={"stage": "anomaly_detection"}):
        result = AnomalyDetector(config).run(features_df)
        m = result["metrics"]

        log_params({"z_threshold": m["z_threshold"], "features": config["anomaly"]["features"]})
        log_metrics({k: m[k] for k in ("state_years_scored", "anomalies_flagged", "anomaly_rate_pct")})
        mlflow.log_dict({"medians": m["feature_medians"], "mads": m["feature_mads"]}, "fitted_statistics.json")

        write_parquet(result["scores"], output_path(config, "predictions", "anomaly_scores"))
        result["audit"] = audit_report(result["scores"])
        path = output_path(config, "reports", "anomaly_audit")
        write_csv_report(result["audit"], path)
        mlflow.log_artifacts(path, "reports/anomaly_audit")
    return result


# ── Console summary ───────────────────────────────────────────────

def _print_summary(forecast: dict, clusters: dict, anomalies: dict, metrics: dict) -> None:
    fm = metrics["forecaster"]
    print("\n" + "=" * 70)
    print("REVENUE FORECASTER — test years (never seen during training)")
    print(f"  {'model':<28}{'growth MAE (pp)':>18}{'revenue MAPE %':>18}")
    for name, m in fm["test"].items():
        print(f"  {name:<28}{m['growth_mae_pct_points']:>18}{m['revenue_mape_pct']:>18}")
    status = "promoted to champion" if fm["promoted_to_champion"] else "kept as challenger"
    print(f"  Registered model version {fm['model_version']} — {status}")
    print(f"  Max feature drift (latest year vs history): {max(fm['drift'].values()):.2f} std")
    print("\nStates with the weakest forecast revenue growth:")
    forecast["forecast"].orderBy("predicted_growth").show(5, truncate=False)

    cm = metrics["clustering"]
    print("=" * 70)
    print(f"PEER CLUSTERS — k={cm['selected_k']}, silhouette={cm['selected_silhouette']}")
    clusters["cluster_profiles"].select("cluster_id", "n_states", "description").show(truncate=False)

    am = metrics["anomaly_detection"]
    print("=" * 70)
    print(f"ANOMALIES — {am['anomalies_flagged']} of {am['state_years_scored']} state-years flagged "
          f"({am['anomaly_rate_pct']}%). Top 10:")
    anomalies["audit"].show(10, truncate=90)
    print("MLflow UI:  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("Dashboard:  uv run python -m dashboard.app")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="US state government finance ML pipeline")
    parser.add_argument("--config", default=None, help="Path to a YAML config (default: config/config.yaml)")
    main(parser.parse_args().config)
