"""
gov_finance_ml/src/models/anomaly_detector.py

Model B — Anomaly Detection for Fraud and Waste
================================================
Insight: Detect statistical outliers in government spending
(Miscellaneous revenue, Interest on debt, Capital outlay spikes)
that may indicate financial mismanagement or data errors requiring audit.

Strategy:
- Isolation Forest   → fast, robust, no distribution assumption
- Autoencoder        → learns "normal" spending patterns, flags reconstruction errors
- Ensemble scoring   → combine both anomaly scores for higher precision
- Results written to Gold layer with audit metadata
"""

from __future__ import annotations

import logging
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.utils.mlflow_utils import (
    log_json_artifact, log_metrics_dict, log_params_flat, managed_run,
    register_model,
)
from src.utils.spark_utils import read_delta, write_delta

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Trains Isolation Forest + Autoencoder models for fiscal anomaly detection.

    Outputs per-record anomaly scores + binary flags to a Gold Delta table.
    """

    ANOMALY_TABLE_SUFFIX = "_anomaly_scores"

    def __init__(self, spark: SparkSession, config: dict) -> None:
        self.spark = spark
        self.cfg = config
        self.model_cfg = config["models"]["anomaly_detector"]
        self.feature_cols = self.model_cfg["features"]
        self.contamination = self.model_cfg["isolation_forest"]["contamination"]

    # ── Public API ─────────────────────────────────

    def run(self, df: Optional[DataFrame] = None) -> DataFrame:
        """Train models, score all records, write anomaly table."""
        if df is None:
            df = read_delta(self.spark, self.cfg["data"]["gold_table"])

        df_train = df.filter(F.col("split") == "train")
        pdf_train = df_train.select(["state", "year"] + self.feature_cols).toPandas()
        pdf_full = df.select(["state", "year"] + self.feature_cols).toPandas()

        X_train, scaler = self._prepare_features(pdf_train)
        X_full = scaler.transform(pdf_full[self.feature_cols].fillna(0))

        # Train models
        iso_model, iso_scores_train = self._train_isolation_forest(X_train)
        ae_model, ae_threshold, ae_scores_train = self._train_autoencoder(X_train)

        # Score full dataset
        iso_scores = iso_model.decision_function(X_full)
        ae_errors = self._autoencoder_errors(ae_model, X_full)
        ensemble_score = self._ensemble_score(iso_scores, ae_errors)

        # Build result DataFrame
        pdf_results = pdf_full[["state", "year"]].copy()
        pdf_results["isolation_forest_score"] = iso_scores
        pdf_results["autoencoder_error"] = ae_errors
        pdf_results["ensemble_anomaly_score"] = ensemble_score

        # Flag anomalies: IF < 0 (anomaly) AND AE error > threshold
        pdf_results["is_anomaly_if"] = (iso_scores < 0).astype(int)
        pdf_results["is_anomaly_ae"] = (ae_errors > ae_threshold).astype(int)
        pdf_results["is_anomaly"] = (
            (pdf_results["is_anomaly_if"] == 1) | (pdf_results["is_anomaly_ae"] == 1)
        ).astype(int)
        pdf_results["anomaly_severity"] = pd.cut(
            ensemble_score,
            bins=[-np.inf, -0.1, 0.0, 0.1, np.inf],
            labels=["critical", "high", "normal", "clean"],
        ).astype(str)

        # MLflow logging
        exp_path = f"{self.cfg['mlflow']['experiment_base']}/anomaly_detection"
        mlflow.set_experiment(exp_path)
        with managed_run("anomaly_detector_training", tags={"model": "ensemble"}) as run:
            log_params_flat({
                "n_estimators": self.model_cfg["isolation_forest"]["n_estimators"],
                "contamination": self.contamination,
                "ae_threshold_pct": self.model_cfg["autoencoder"]["threshold_percentile"],
                "feature_count": len(self.feature_cols),
            })
            anomaly_rate = pdf_results["is_anomaly"].mean()
            log_metrics_dict({
                "anomaly_rate": float(anomaly_rate),
                "ae_reconstruction_threshold": float(ae_threshold),
                "n_records_scored": float(len(pdf_results)),
                "n_anomalies_detected": float(pdf_results["is_anomaly"].sum()),
            })
            mlflow.sklearn.log_model(iso_model, "isolation_forest")
            log_json_artifact(
                {"feature_cols": self.feature_cols, "threshold": ae_threshold},
                "anomaly_config.json",
            )
            registered = f"{self.cfg['mlflow']['registered_model_prefix']}.anomaly_detector"
            version = register_model(
                run_id=run.info.run_id,
                artifact_path="isolation_forest",
                registered_name=registered,
                alias="champion",
                description=f"Fiscal anomaly detector | anomaly_rate={anomaly_rate:.3f}",
            )
            logger.info("Anomaly model registered: %s v%s", registered, version)

        # Persist scores to Delta
        anomaly_table = self.cfg["data"]["gold_table"] + self.ANOMALY_TABLE_SUFFIX
        result_spark = self.spark.createDataFrame(pdf_results)
        write_delta(result_spark, anomaly_table, mode="overwrite", partition_by=["state"])
        logger.info("Anomaly scores written → %s", anomaly_table)

        return result_spark

    def get_audit_report(self, df: Optional[DataFrame] = None) -> pd.DataFrame:
        """Return top anomalous records sorted by severity."""
        anomaly_table = self.cfg["data"]["gold_table"] + self.ANOMALY_TABLE_SUFFIX
        adf = df or read_delta(self.spark, anomaly_table)
        return (
            adf.filter(F.col("is_anomaly") == 1)
            .orderBy(F.col("ensemble_anomaly_score").asc())
            .toPandas()
        )

    # ── Feature preparation ───────────────────────

    def _prepare_features(self, pdf: pd.DataFrame):
        X = pdf[self.feature_cols].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler

    # ── Isolation Forest ──────────────────────────

    def _train_isolation_forest(self, X: np.ndarray):
        cfg = self.model_cfg["isolation_forest"]
        model = IsolationForest(
            n_estimators=cfg["n_estimators"],
            contamination=cfg["contamination"],
            random_state=cfg["random_state"],
            n_jobs=-1,
        )
        model.fit(X)
        scores = model.decision_function(X)  # higher = more normal
        logger.info(
            "Isolation Forest trained | anomaly rate=%.3f",
            (scores < 0).mean(),
        )
        return model, scores

    # ── Autoencoder ───────────────────────────────

    def _train_autoencoder(self, X: np.ndarray):
        """
        Lightweight NumPy autoencoder (no TF dependency required for simple use).
        For production, swap with a TensorFlow/PyTorch version via mlflow.pytorch.
        """
        from sklearn.neural_network import MLPRegressor

        cfg = self.model_cfg["autoencoder"]
        n_features = X.shape[1]
        hidden_sizes = tuple(cfg["encoding_dim"])

        # Encoder + decoder via symmetric MLP
        ae = MLPRegressor(
            hidden_layer_sizes=(*hidden_sizes, *hidden_sizes[::-1]),
            activation="relu",
            max_iter=cfg["epochs"],
            batch_size=cfg["batch_size"],
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        ae.fit(X, X)

        reconstructed = ae.predict(X)
        errors = np.mean((X - reconstructed) ** 2, axis=1)

        threshold = np.percentile(errors, cfg["threshold_percentile"])
        logger.info(
            "Autoencoder trained | threshold=%.4f | anomaly rate=%.3f",
            threshold, (errors > threshold).mean(),
        )
        return ae, threshold, errors

    def _autoencoder_errors(self, model, X: np.ndarray) -> np.ndarray:
        reconstructed = model.predict(X)
        return np.mean((X - reconstructed) ** 2, axis=1)

    # ── Ensemble ──────────────────────────────────

    def _ensemble_score(
        self, iso_scores: np.ndarray, ae_errors: np.ndarray
    ) -> np.ndarray:
        """
        Combine normalised IF score (higher=normal) and AE error (lower=normal).
        Final ensemble score: lower = more anomalous.
        """
        iso_norm = (iso_scores - iso_scores.mean()) / (iso_scores.std() + 1e-8)
        ae_norm = -((ae_errors - ae_errors.mean()) / (ae_errors.std() + 1e-8))  # flip
        return 0.5 * iso_norm + 0.5 * ae_norm
