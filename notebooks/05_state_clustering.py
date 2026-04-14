# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Policy-Driven State Clustering
# MAGIC **Model C: K-Means with Optimal K Selection**
# MAGIC
# MAGIC Groups states by financial DNA so policymakers can benchmark
# MAGIC against true peers. Produces cluster archetypes and a peer-state
# MAGIC recommendation engine.

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Repos/gov_finance_ml")

from src.utils.spark_utils import get_spark, load_config, read_delta
from src.models.state_clustering import StateClusteringModel, CLUSTER_ARCHETYPES
from pyspark.sql import functions as F
import pandas as pd

spark = get_spark()
cfg = load_config()

# COMMAND ----------
# MAGIC %md ## 5.1 — Train Clustering Model

# COMMAND ----------

gold_df = read_delta(spark, cfg["data"]["gold_table"])
clustering_model = StateClusteringModel(spark, cfg)
cluster_df = clustering_model.run(df=gold_df)

display(
    cluster_df.groupBy("cluster_id", "cluster_archetype")
    .agg(F.count("*").alias("n_records"), F.countDistinct("state").alias("n_states"))
    .orderBy("cluster_id")
)

# COMMAND ----------
# MAGIC %md ## 5.2 — Cluster Distribution Over Time

# COMMAND ----------

display(
    cluster_df.groupBy("year", "cluster_id")
    .count()
    .orderBy("year", "cluster_id")
)

# COMMAND ----------
# MAGIC %md ## 5.3 — Cluster Profiles (Latest Year)

# COMMAND ----------

max_year = gold_df.agg(F.max("year")).collect()[0][0]
cluster_labels = read_delta(
    spark,
    cfg["data"]["gold_table"] + "_cluster_labels",
)
feature_cols = cfg["models"]["clustering"]["features"]
profile_df = (
    gold_df.filter(F.col("year") == max_year)
    .join(cluster_labels.select("state", "cluster_id", "cluster_archetype"), on="state", how="left")
    .groupBy("cluster_id", "cluster_archetype")
    .agg(*[F.avg(c).alias(c) for c in feature_cols])
    .orderBy("cluster_id")
)
display(profile_df)

# COMMAND ----------
# MAGIC %md ## 5.4 — Peer State Lookup

# COMMAND ----------

# Find peers for any state in the latest year
TARGET_STATE = "TEXAS"
TARGET_YEAR = max_year

print(f"\n=== Peer states for {TARGET_STATE} ({TARGET_YEAR}) ===")
peers = clustering_model.get_peer_states(state=TARGET_STATE, year=TARGET_YEAR, top_n=8)
print(peers.to_string(index=False))

# COMMAND ----------
# MAGIC %md ## 5.5 — State Cluster Trajectory (Has a state changed clusters?)

# COMMAND ----------

# Track cluster ID per state across years
display(
    cluster_df.groupBy("state")
    .agg(F.countDistinct("cluster_id").alias("n_cluster_changes"))
    .filter(F.col("n_cluster_changes") > 1)
    .orderBy(F.col("n_cluster_changes").desc())
    .limit(15)
)

# COMMAND ----------
# MAGIC %md ## 5.6 — Silhouette Score Trend

# COMMAND ----------

import mlflow
exp = mlflow.get_experiment_by_name(f"{cfg['mlflow']['experiment_base']}/state_clustering")
if exp:
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.silhouette_score DESC"],
    )
    if not runs.empty:
        sil_df = runs[["params.optimal_k", "metrics.silhouette_score", "metrics.inertia"]].copy()
        display(spark.createDataFrame(sil_df.fillna("")))

# COMMAND ----------
# MAGIC %md
# MAGIC ## ✅ State Clustering Complete
# MAGIC
# MAGIC Cluster table: `gov_finance.gold.state_finances_features_cluster_labels`
# MAGIC
# MAGIC Key use cases:
# MAGIC - **Federal policy targeting**: apply different interventions per cluster
# MAGIC - **Peer benchmarking**: compare a state to its fiscal twins
# MAGIC - **Early warning**: detect states migrating to high-debt clusters
# MAGIC
# MAGIC Proceed to **06_resource_optimization**
