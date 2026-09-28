"""
Model B — Peer-state clustering (Spark ML KMeans).

Business question: "Which states have a similar financial structure, and
where does a state spend noticeably more or less than its peers?"
Comparing Wyoming with California is not useful; comparing a state with
peers that fund themselves and spend in a similar way is.

How it works
------------
1. One profile row per state: the average of each feature over the most
   recent N years (smooths out one-off years).
2. Features are ratios/shares (budget STRUCTURE, not size), standardised
   so each feature counts equally in the distance calculation.
3. KMeans is trained for each k in k_range; the k with the best silhouette
   score (how well-separated the clusters are, -1..1) is kept.
4. Each cluster gets an automatic description from the features where its
   average is furthest from the national average (in standard deviations).
5. Peer benchmark: for every state and feature, the gap versus the median
   of its cluster, e.g. welfare_share_vs_peers = +0.04 means the state
   spends 4 percentage points more of its budget on welfare than its peers.
"""

from __future__ import annotations

import logging

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


class StateClusteringModel:
    def __init__(self, config: dict) -> None:
        cfg = config["clustering"]
        self.feature_cols = cfg["features"]
        self.k_range = cfg["k_range"]
        self.profile_years = cfg["profile_years"]
        self.seed = cfg["seed"]

    # ── Public API ──────────────────────────────────────────────

    def run(self, features_df: DataFrame) -> dict:
        profiles = self.build_state_profiles(features_df).cache()
        n_states = profiles.count()
        logger.info("Clustering %d state profiles", n_states)

        silhouette_by_k = {}
        best = None
        for k in self.k_range:
            if k >= n_states:
                break
            model = self._build_pipeline(k).fit(profiles)
            predictions = model.transform(profiles)
            silhouette = ClusteringEvaluator(
                featuresCol="scaled_features", predictionCol="cluster_id"
            ).evaluate(predictions)
            silhouette_by_k[k] = round(silhouette, 4)
            logger.info("k=%d -> silhouette %.4f", k, silhouette)
            if best is None or silhouette > best[1]:
                best = (k, silhouette, model, predictions)

        best_k, best_silhouette, model, predictions = best
        logger.info("Selected k=%d (silhouette %.4f)", best_k, best_silhouette)

        profiles.unpersist()
        return {
            "model": model,
            "state_clusters": self._peer_benchmark(predictions),
            "cluster_profiles": self._describe_clusters(predictions),
            "metrics": {
                "profile_years": self.profile_years,
                "silhouette_by_k": silhouette_by_k,
                "selected_k": best_k,
                "selected_silhouette": round(best_silhouette, 4),
            },
        }

    def build_state_profiles(self, features_df: DataFrame) -> DataFrame:
        """Average each feature over each state's most recent `profile_years` years."""
        max_year = features_df.agg(F.max("year")).first()[0]
        first_year = max_year - self.profile_years + 1
        return (
            features_df.filter(F.col("year") >= first_year)
            .groupBy("state")
            .agg(*[F.avg(c).alias(c) for c in self.feature_cols])
            .dropna()
        )

    # ── Internals ───────────────────────────────────────────────

    def _build_pipeline(self, k: int) -> Pipeline:
        return Pipeline(stages=[
            VectorAssembler(inputCols=self.feature_cols, outputCol="features"),
            StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True),
            KMeans(k=k, featuresCol="scaled_features", predictionCol="cluster_id", seed=self.seed),
        ])

    def _describe_clusters(self, predictions: DataFrame) -> DataFrame:
        """One row per cluster: size, member states, feature averages and a text description."""
        scaled = predictions.withColumn("z", vector_to_array("scaled_features"))
        z_avgs = [F.avg(F.col("z")[i]).alias(f"z_{c}") for i, c in enumerate(self.feature_cols)]
        raw_avgs = [F.round(F.avg(c), 4).alias(f"avg_{c}") for c in self.feature_cols]

        summary = (
            scaled.groupBy("cluster_id")
            .agg(
                F.count("*").alias("n_states"),
                F.concat_ws(", ", F.sort_array(F.collect_list("state"))).alias("states"),
                *raw_avgs,
                *z_avgs,
            )
            .orderBy("cluster_id")
        )

        # Describe each cluster by its 3 most distinctive features (largest |z|)
        descriptions = {}
        for row in summary.collect():
            z_scores = {c: row[f"z_{c}"] for c in self.feature_cols}
            top = sorted(z_scores.items(), key=lambda kv: -abs(kv[1]))[:3]
            descriptions[row["cluster_id"]] = ", ".join(
                f"{'high' if z > 0 else 'low'} {name}" for name, z in top
            )
        description_df = summary.sparkSession.createDataFrame(
            list(descriptions.items()), ["cluster_id", "description"]
        )

        return (
            summary.join(description_df, "cluster_id")
            .select("cluster_id", "description", "n_states", "states", *[f"avg_{c}" for c in self.feature_cols])
            .orderBy("cluster_id")
        )

    def _peer_benchmark(self, predictions: DataFrame) -> DataFrame:
        """Each state's feature values minus the median of its cluster."""
        medians = predictions.groupBy("cluster_id").agg(
            *[F.percentile_approx(c, 0.5).alias(f"peer_median_{c}") for c in self.feature_cols]
        )
        joined = predictions.join(medians, "cluster_id")
        return joined.select(
            "state",
            "cluster_id",
            *[F.round(c, 4).alias(c) for c in self.feature_cols],
            *[
                F.round(F.col(c) - F.col(f"peer_median_{c}"), 4).alias(f"{c}_vs_peers")
                for c in self.feature_cols
            ],
        ).orderBy("cluster_id", "state")


def get_peer_states(state_clusters: DataFrame, state: str) -> list[str]:
    """States in the same cluster as `state` (excluding itself)."""
    row = state_clusters.filter(F.col("state") == state.upper()).select("cluster_id").first()
    if row is None:
        raise ValueError(f"Unknown state: {state}")
    peers = state_clusters.filter(
        (F.col("cluster_id") == row["cluster_id"]) & (F.col("state") != state.upper())
    )
    return sorted(r["state"] for r in peers.select("state").collect())
