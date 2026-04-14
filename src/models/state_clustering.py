"""
gov_finance_ml/src/models/state_clustering.py

Model C — Policy-Driven State Clustering
=========================================
Insight: Group states by financial "DNA" so policymakers can benchmark
against true peers rather than applying one-size-fits-all federal policy.

Strategy:
- PySpark ML KMeans (native distributed)
- Silhouette score + Elbow method for optimal K selection
- Cluster profiling → human-readable archetypes
- Peer recommendation engine
"""

from __future__ import annotations

import logging
from typing import Optional

import mlflow
import mlflow.spark
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.clustering import BisectingKMeans, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import Imputer, MinMaxScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.utils.mlflow_utils import (
    log_json_artifact, log_metrics_dict, log_params_flat, managed_run,
    register_model,
)
from src.utils.spark_utils import read_delta, write_delta

logger = logging.getLogger(__name__)

# Human-readable cluster archetypes (post hoc, updated after inspection)
CLUSTER_ARCHETYPES = {
    0: "High-Revenue Industrial States",
    1: "Rural Intergovernmental-Dependent States",
    2: "Balanced Mid-Tier States",
    3: "Debt-Heavy Infrastructure Investors",
    4: "Welfare-Prioritising Urban States",
    5: "Low-Tax Small Government States",
    6: "Energy-Rich License-Tax States",
    7: "Mixed Economy Transition States",
}


class StateClusteringModel:
    """
    Trains KMeans clustering on state financial features.
    Supports K selection via silhouette + elbow, and produces:
    - Cluster labels per (state, year)
    - Cluster profiles (centroid descriptions)
    - Peer-state recommendation table
    """

    CLUSTER_TABLE_SUFFIX = "_cluster_labels"

    def __init__(self, spark: SparkSession, config: dict) -> None:
        self.spark = spark
        self.cfg = config
        self.model_cfg = config["models"]["clustering"]
        self.feature_cols = self.model_cfg["features"]
        self.k_range = self.model_cfg["k_range"]

    # ── Public API ─────────────────────────────────

    def run(self, df: Optional[DataFrame] = None) -> DataFrame:
        """Select optimal K, train final model, write labels."""
        if df is None:
            df = read_delta(self.spark, self.cfg["data"]["gold_table"])

        df_train = df.filter(F.col("split") == "train")
        pipeline_model, df_prep = self._build_and_fit_pipeline(df_train)

        # Select optimal K
        exp_path = f"{self.cfg['mlflow']['experiment_base']}/state_clustering"
        mlflow.set_experiment(exp_path)

        silhouette_scores = {}
        inertia_scores = {}

        for k in self.k_range:
            sil, inertia, _ = self._evaluate_k(df_prep, k)
            silhouette_scores[k] = sil
            inertia_scores[k] = inertia
            logger.info("K=%d | Silhouette=%.4f | Inertia=%.2f", k, sil, inertia)

        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
        logger.info("Optimal K selected: %d (silhouette=%.4f)", optimal_k, silhouette_scores[optimal_k])

        # Train final model with optimal K
        with managed_run(
            f"kmeans_k{optimal_k}_final",
            tags={"model": "kmeans", "k": str(optimal_k)},
        ) as run:
            log_params_flat({
                "optimal_k": optimal_k,
                "k_range": str(self.k_range),
                "n_features": len(self.feature_cols),
                "random_state": self.model_cfg["random_state"],
            })
            log_metrics_dict({
                "silhouette_score": silhouette_scores[optimal_k],
                "inertia": inertia_scores[optimal_k],
            })
            log_json_artifact(
                {"silhouette_by_k": silhouette_scores, "inertia_by_k": inertia_scores},
                "k_selection_metrics.json",
            )

            final_sil, final_inertia, predictions = self._evaluate_k(df_prep, optimal_k)
            cluster_profiles = self._profile_clusters(predictions, optimal_k)
            log_json_artifact(cluster_profiles, "cluster_profiles.json")

            # Log pipeline model
            mlflow.spark.log_model(pipeline_model, "preprocessing_pipeline")

            registered = f"{self.cfg['mlflow']['registered_model_prefix']}.state_clustering"
            register_model(
                run_id=run.info.run_id,
                artifact_path="preprocessing_pipeline",
                registered_name=registered,
                alias="champion",
                description=f"State clustering K={optimal_k} | silhouette={final_sil:.4f}",
            )

        # Attach cluster labels back to full dataset
        full_prep_df = pipeline_model.transform(df)
        kmeans = KMeans(
            k=optimal_k,
            featuresCol="features",
            predictionCol="cluster_id",
            seed=self.model_cfg["random_state"],
        )
        kmeans_model = kmeans.fit(df_prep)
        labelled_df = kmeans_model.transform(full_prep_df)

        # Add archetype names and peer context
        archetype_map = {k: CLUSTER_ARCHETYPES.get(k, f"Cluster_{k}") for k in range(optimal_k)}
        archetype_udf = F.udf(lambda cid: archetype_map.get(cid, f"Cluster_{cid}"))
        labelled_df = labelled_df.withColumn("cluster_archetype", archetype_udf(F.col("cluster_id")))

        # Write to Delta
        cluster_table = self.cfg["data"]["gold_table"] + self.CLUSTER_TABLE_SUFFIX
        output_df = labelled_df.select(
            "state", "year", "cluster_id", "cluster_archetype", "split"
        )
        write_delta(output_df, cluster_table, mode="overwrite", partition_by=["state"])
        logger.info("Cluster labels written → %s", cluster_table)

        return output_df

    def get_peer_states(self, state: str, year: int, top_n: int = 5) -> pd.DataFrame:
        """Return states in the same cluster as a given (state, year)."""
        cluster_table = self.cfg["data"]["gold_table"] + self.CLUSTER_TABLE_SUFFIX
        df = read_delta(self.spark, cluster_table)

        target = df.filter(
            (F.col("state") == state.upper()) & (F.col("year") == year)
        ).first()
        if target is None:
            raise ValueError(f"State '{state}' not found for year {year}")

        cluster_id = target["cluster_id"]
        peers = (
            df.filter(
                (F.col("cluster_id") == cluster_id)
                & (F.col("state") != state.upper())
                & (F.col("year") == year)
            )
            .select("state", "cluster_archetype")
            .distinct()
            .limit(top_n)
            .toPandas()
        )
        logger.info(
            "Peers of %s (%d): %s",
            state, year, peers["state"].tolist(),
        )
        return peers

    # ── Pipeline & evaluation ─────────────────────

    def _build_and_fit_pipeline(self, df: DataFrame):
        imputer = Imputer(
            inputCols=self.feature_cols,
            outputCols=[f"{c}_imp" for c in self.feature_cols],
            strategy="median",
        )
        assembler = VectorAssembler(
            inputCols=[f"{c}_imp" for c in self.feature_cols],
            outputCol="features_raw",
            handleInvalid="skip",
        )
        scaler = MinMaxScaler(inputCol="features_raw", outputCol="features")
        pipeline = Pipeline(stages=[imputer, assembler, scaler])
        pipeline_model = pipeline.fit(df)
        transformed = pipeline_model.transform(df)
        return pipeline_model, transformed

    def _evaluate_k(self, df_prep: DataFrame, k: int):
        kmeans = KMeans(
            k=k,
            featuresCol="features",
            predictionCol="cluster_id",
            seed=self.model_cfg["random_state"],
            maxIter=100,
        )
        model = kmeans.fit(df_prep)
        predictions = model.transform(df_prep)

        evaluator = ClusteringEvaluator(
            featuresCol="features", predictionCol="cluster_id", metricName="silhouette"
        )
        silhouette = evaluator.evaluate(predictions)
        inertia = float(model.summary.trainingCost)
        return silhouette, inertia, predictions

    # ── Cluster profiling ─────────────────────────

    def _profile_clusters(self, predictions: DataFrame, k: int) -> dict:
        """Compute mean feature values per cluster for interpretability."""
        agg_cols = [F.avg(c).alias(c) for c in self.feature_cols if c in predictions.columns]
        profile_pdf = (
            predictions.groupBy("cluster_id")
            .agg(*agg_cols, F.count("*").alias("n_records"))
            .orderBy("cluster_id")
            .toPandas()
        )

        profiles = {}
        for _, row in profile_pdf.iterrows():
            cid = int(row["cluster_id"])
            profiles[cid] = {
                "archetype": CLUSTER_ARCHETYPES.get(cid, f"Cluster_{cid}"),
                "n_records": int(row["n_records"]),
                "top_features": {
                    col: round(float(row[col]), 4)
                    for col in self.feature_cols
                    if col in row and pd.notna(row[col])
                },
            }
        return profiles
