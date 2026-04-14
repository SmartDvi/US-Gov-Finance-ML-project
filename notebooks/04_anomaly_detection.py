# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Anomaly Detection for Fiscal Fraud & Waste
# MAGIC **Model B: Isolation Forest + Autoencoder Ensemble**
# MAGIC
# MAGIC Detects anomalous spending patterns in:
# MAGIC - Miscellaneous general revenue (opacity proxy)
# MAGIC - Capital outlay spikes
# MAGIC - Interest on general debt
# MAGIC - Revenue vs expenditure divergence
# MAGIC
# MAGIC Results are written to `gold.state_finances_features_anomaly_scores`.

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Repos/gov_finance_ml")

from src.utils.spark_utils import get_spark, load_config, read_delta
from src.models.anomaly_detector import AnomalyDetector
from pyspark.sql import functions as F

spark = get_spark()
cfg = load_config()

# COMMAND ----------
# MAGIC %md ## 4.1 — Train Anomaly Detectors

# COMMAND ----------

detector = AnomalyDetector(spark, cfg)
anomaly_df = detector.run()
print(f"Anomaly scores generated for {anomaly_df.count():,} records")

# COMMAND ----------
# MAGIC %md ## 4.2 — Anomaly Distribution

# COMMAND ----------

display(
    anomaly_df.groupBy("anomaly_severity")
    .agg(F.count("*").alias("count"))
    .orderBy("count")
)

# COMMAND ----------

# Anomaly rate per state
display(
    anomaly_df.groupBy("state")
    .agg(
        F.avg("is_anomaly").alias("anomaly_rate"),
        F.sum("is_anomaly").alias("n_anomalies"),
        F.avg("ensemble_anomaly_score").alias("avg_anomaly_score"),
    )
    .orderBy(F.col("anomaly_rate").desc())
    .limit(15)
)

# COMMAND ----------
# MAGIC %md ## 4.3 — Audit Report (Critical Anomalies)

# COMMAND ----------

audit_pdf = detector.get_audit_report(anomaly_df)
print(f"Critical/High anomalies requiring audit: {len(audit_pdf):,}")
display(spark.createDataFrame(audit_pdf.head(20)))

# COMMAND ----------
# MAGIC %md ## 4.4 — Join Anomaly Flags to Gold Data

# COMMAND ----------

gold_df = read_delta(spark, cfg["data"]["gold_table"])
enriched = gold_df.join(
    anomaly_df.select("state", "year", "is_anomaly", "anomaly_severity", "ensemble_anomaly_score"),
    on=["state", "year"],
    how="left",
)

# Anomaly years per state
display(
    enriched.filter(F.col("is_anomaly") == 1)
    .select("state", "year", "anomaly_severity", "ensemble_anomaly_score",
            "totals_capital_outlay", "details_interest_on_general_debt",
            "details_miscellaneous_general_revenue")
    .orderBy("ensemble_anomaly_score")
    .limit(20)
)

# COMMAND ----------
# MAGIC %md ## 4.5 — Isolation Forest Feature Importance (SHAP proxy)

# COMMAND ----------

from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

feature_cols = cfg["models"]["anomaly_detector"]["features"]
pdf = gold_df.filter(F.col("split") == "train").select(feature_cols).toPandas().fillna(0)

iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
iso.fit(pdf.values)

# Estimate feature importance via mean depth
scores = iso.decision_function(pdf.values)
feat_contributions = []
for i, col in enumerate(feature_cols):
    perturbed = pdf.copy()
    perturbed[col] = 0
    perturbed_scores = iso.decision_function(perturbed.values)
    contribution = float(np.abs(scores - perturbed_scores).mean())
    feat_contributions.append({"feature": col, "importance": contribution})

feat_df = pd.DataFrame(feat_contributions).sort_values("importance", ascending=False)
display(spark.createDataFrame(feat_df))

# COMMAND ----------
# MAGIC %md
# MAGIC ## ✅ Anomaly Detection Complete
# MAGIC
# MAGIC Scores table: `gov_finance.gold.state_finances_features_anomaly_scores`
# MAGIC
# MAGIC Severity legend:
# MAGIC | Severity | Action |
# MAGIC |----------|--------|
# MAGIC | `critical` | Flag for immediate audit |
# MAGIC | `high` | Schedule quarterly review |
# MAGIC | `normal` | Monitor |
# MAGIC | `clean` | No action |
# MAGIC
# MAGIC Proceed to **05_state_clustering**
