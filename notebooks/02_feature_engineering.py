# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Feature Engineering (Silver → Gold)
# MAGIC
# MAGIC Builds the analytical Gold layer with:
# MAGIC - Financial ratio features (policy-insight driven)
# MAGIC - Year-over-year growth rates
# MAGIC - Lag and rolling window features (for time-series models)
# MAGIC - Composite indices: Fiscal Health Index, Social Investment Score
# MAGIC - Binary sustainability flags

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Repos/gov_finance_ml")

from src.utils.spark_utils import get_spark, load_config, read_delta
from src.features.feature_engineering import FeatureEngineer
from pyspark.sql import functions as F

spark = get_spark()
cfg = load_config()

# COMMAND ----------
# MAGIC %md ## 2.1 — Run Feature Engineering

# COMMAND ----------

fe = FeatureEngineer(spark, cfg)
gold_df = fe.run()

print(f"Gold columns: {len(gold_df.columns)}")
print(f"Gold rows: {gold_df.count():,}")
display(gold_df.limit(5))

# COMMAND ----------
# MAGIC %md ## 2.2 — Feature Distributions

# COMMAND ----------

# Fiscal Health Index distribution by state (latest year)
max_year = gold_df.agg(F.max("year")).collect()[0][0]
display(
    gold_df.filter(F.col("year") == max_year)
    .select("state", "fiscal_health_index", "debt_to_revenue_ratio",
            "social_investment_score", "welfare_expenditure_share")
    .orderBy(F.col("fiscal_health_index").desc())
)

# COMMAND ----------

# Flag counts
display(
    gold_df.agg(
        F.sum("debt_critical").alias("debt_critical_count"),
        F.sum("expenditure_over_revenue").alias("exp_over_rev_count"),
        F.sum("high_interest_burden").alias("high_interest_count"),
        F.count("*").alias("total_records"),
    )
)

# COMMAND ----------
# MAGIC %md ## 2.3 — Feature Correlation Heatmap (Pandas)

# COMMAND ----------

ratio_cols = [
    "debt_to_revenue_ratio", "expenditure_to_revenue_ratio",
    "welfare_expenditure_share", "highway_expenditure_share",
    "police_vs_parks_ratio", "interest_burden",
    "intergovernmental_dependency", "fiscal_health_index",
    "social_investment_score",
]
import pandas as pd
pdf_corr = gold_df.select(ratio_cols).toPandas()
corr_matrix = pdf_corr.corr().round(3)
display(spark.createDataFrame(corr_matrix.reset_index().rename(columns={"index": "feature"})))

# COMMAND ----------
# MAGIC %md ## 2.4 — Build PySpark ML Pipeline (for clustering / forecasting)

# COMMAND ----------

clustering_pipeline = fe.build_ml_pipeline(
    feature_cols=fe.clustering_features,
    scale="minmax",
)
print("Clustering ML pipeline stages:")
for stage in clustering_pipeline.getStages():
    print(f"  - {type(stage).__name__}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## ✅ Feature Engineering Complete
# MAGIC
# MAGIC | Layer | Table | Features Added |
# MAGIC |-------|-------|----------------|
# MAGIC | Gold | `gov_finance.gold.state_finances_features` | ~60 features |
# MAGIC
# MAGIC Key composite scores:
# MAGIC - `fiscal_health_index` ∈ [0,1] — higher = healthier
# MAGIC - `social_investment_score` ∈ [0,1] — welfare + parks balance
# MAGIC - `debt_critical`, `expenditure_over_revenue`, `high_interest_burden` — binary flags
# MAGIC
# MAGIC Proceed to **03_fiscal_forecasting**
