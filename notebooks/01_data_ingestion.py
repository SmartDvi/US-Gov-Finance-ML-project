# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Data Ingestion & Exploration
# MAGIC **Government Finance ML Pipeline**
# MAGIC
# MAGIC This notebook:
# MAGIC - Ingests the raw state finance CSV into the Bronze Delta table
# MAGIC - Runs Silver preprocessing (cleaning, imputation, quality scoring)
# MAGIC - Profiles the cleaned data

# COMMAND ----------
# MAGIC %md ## Setup

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Repos/gov_finance_ml")

import logging
logging.basicConfig(level=logging.INFO)

import yaml
from src.utils.spark_utils import get_spark, load_config
from src.data.ingestion import BronzeIngestion
from src.data.preprocessing import SilverProcessor

spark = get_spark()
cfg = load_config()
print(f"Project: {cfg['project']['name']} v{cfg['project']['version']}")

# COMMAND ----------
# MAGIC %md ## 1.1 — Create Catalog / Schema (run once)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS gov_finance;
# MAGIC CREATE SCHEMA IF NOT EXISTS gov_finance.bronze;
# MAGIC CREATE SCHEMA IF NOT EXISTS gov_finance.silver;
# MAGIC CREATE SCHEMA IF NOT EXISTS gov_finance.gold;

# COMMAND ----------
# MAGIC %md ## 1.2 — Ingest Raw CSV → Bronze

# COMMAND ----------

# Upload your CSV to a DBFS path or Unity Catalog Volume first, then set this path
RAW_CSV_PATH = "dbfs:/FileStore/gov_finance/state_finances_raw.csv"
# For UC volumes: "/Volumes/gov_finance/raw_data/state_finances_raw.csv"

bronze_ingestor = BronzeIngestion(spark, cfg)
bronze_df = bronze_ingestor.ingest_csv(
    path=RAW_CSV_PATH,
    mode="overwrite",
    header=True,
    delimiter=",",
)
print(f"Bronze rows: {bronze_df.count():,}")
display(bronze_df.limit(10))

# COMMAND ----------
# MAGIC %md ## 1.3 — Bronze Quality Check

# COMMAND ----------

from src.utils.spark_utils import null_report, describe_numeric

print("=== NULL REPORT ===")
display(null_report(bronze_df))

print("\n=== DESCRIPTIVE STATS ===")
display(describe_numeric(bronze_df))

# COMMAND ----------
# MAGIC %md ## 1.4 — Silver Processing

# COMMAND ----------

silver_proc = SilverProcessor(spark, cfg)
silver_df = silver_proc.run(input_df=bronze_df)

print("=== SILVER QUALITY REPORT ===")
import json
report = silver_proc.quality_report(silver_df)
print(json.dumps(report, indent=2))

# COMMAND ----------
# MAGIC %md ## 1.5 — Exploratory Analysis

# COMMAND ----------

from pyspark.sql import functions as F

# Revenue trend by year
revenue_trend = (
    silver_df.groupBy("year")
    .agg(
        F.sum("totals_revenue").alias("total_revenue"),
        F.avg("totals_revenue").alias("avg_revenue"),
        F.sum("totals_expenditure").alias("total_expenditure"),
        F.sum("totals_debt_at_end_of_fiscal_year").alias("total_debt"),
    )
    .orderBy("year")
)
display(revenue_trend)

# COMMAND ----------

# Top 10 states by debt/revenue ratio in latest year
from pyspark.sql.window import Window
max_year = silver_df.agg(F.max("year")).collect()[0][0]
top_debt_states = (
    silver_df.filter(F.col("year") == max_year)
    .withColumn(
        "debt_ratio",
        F.col("totals_debt_at_end_of_fiscal_year") / (F.col("totals_revenue") + 1),
    )
    .select("state", "year", "totals_revenue", "totals_debt_at_end_of_fiscal_year", "debt_ratio")
    .orderBy(F.col("debt_ratio").desc())
    .limit(10)
)
display(top_debt_states)

# COMMAND ----------

# Data quality score distribution
display(
    silver_df.groupBy("split")
    .agg(
        F.avg("data_quality_score").alias("avg_dqs"),
        F.min("data_quality_score").alias("min_dqs"),
        F.max("data_quality_score").alias("max_dqs"),
        F.count("*").alias("n_rows"),
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## ✅ Ingestion & Preprocessing Complete
# MAGIC
# MAGIC | Layer | Table | Status |
# MAGIC |-------|-------|--------|
# MAGIC | Bronze | `gov_finance.bronze.state_finances_raw` | ✓ |
# MAGIC | Silver | `gov_finance.silver.state_finances` | ✓ |
# MAGIC
# MAGIC Proceed to **02_feature_engineering**
