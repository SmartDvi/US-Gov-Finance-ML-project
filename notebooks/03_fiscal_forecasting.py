# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Fiscal Health Forecasting
# MAGIC **Model A: Prophet + XGBoost Revenue / Expenditure Forecasting**
# MAGIC
# MAGIC Trains both Prophet and XGBoost models across a hyperparameter grid.
# MAGIC All runs are tracked in MLflow. The best model is registered as **champion**
# MAGIC in the Unity Catalog Model Registry.

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Repos/gov_finance_ml")

from src.utils.spark_utils import get_spark, load_config, read_delta
from src.models.fiscal_forecaster import FiscalForecaster
from pyspark.sql import functions as F

spark = get_spark()
cfg = load_config()

# COMMAND ----------
# MAGIC %md ## 3.1 — Load Gold Data

# COMMAND ----------

gold_df = read_delta(spark, cfg["data"]["gold_table"])
print(f"Gold rows: {gold_df.count():,}")
print(f"Train rows: {gold_df.filter(F.col('split')=='train').count():,}")
print(f"Test rows:  {gold_df.filter(F.col('split')=='test').count():,}")

# COMMAND ----------
# MAGIC %md ## 3.2 — Train Forecasters (Revenue + Expenditure)

# COMMAND ----------

forecaster = FiscalForecaster(spark, cfg)
results = forecaster.run(df=gold_df)

for target, metrics in results.items():
    print(f"\n{'='*50}")
    print(f"Target: {target}")
    print(f"  Best RMSE : {metrics['best_rmse']:,.2f}")
    print(f"  Best run  : {metrics['best_run_id']}")

# COMMAND ----------
# MAGIC %md ## 3.3 — Per-State XGBoost Training (Distributed via Pandas UDF)

# COMMAND ----------
# MAGIC %md
# MAGIC This scales the model to train independently per state using PySpark's
# MAGIC Pandas UDF, distributing work across the cluster.

# COMMAND ----------

state_metrics = forecaster.train_per_state_xgb_udf(gold_df, target="totals_revenue")
display(state_metrics.orderBy("rmse"))

# COMMAND ----------
# MAGIC %md ## 3.4 — Forecast Visualisation (Top 5 States by Revenue)

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# National-level forecast
pdf = (
    gold_df.groupBy("year")
    .agg(F.sum("totals_revenue").alias("y"))
    .orderBy("year")
    .toPandas()
)
pdf["ds"] = pd.to_datetime(pdf["year"], format="%Y")
train_pdf = pdf[pdf["year"] <= cfg["data"]["train_cutoff_year"]]

model = Prophet(changepoint_prior_scale=0.05, seasonality_mode="additive",
                yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
model.fit(train_pdf[["ds","y"]])
future = model.make_future_dataframe(periods=cfg["models"]["fiscal_forecaster"]["horizon_years"], freq="Y")
forecast = model.predict(future)

fig = model.plot(forecast)
plt.title("National Revenue Forecast — Prophet")
plt.tight_layout()
display(fig)

# COMMAND ----------
# MAGIC %md ## 3.5 — MLflow Experiment Summary

# COMMAND ----------

import mlflow
experiment_name = f"{cfg['mlflow']['experiment_base']}/fiscal_forecasting"
exp = mlflow.get_experiment_by_name(experiment_name)
if exp:
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=10,
    )
    display(spark.createDataFrame(runs[[
        "run_id", "tags.model", "tags.target",
        "params.n_estimators", "params.max_depth", "params.learning_rate",
        "metrics.rmse", "metrics.mae", "metrics.r2", "metrics.mape"
    ]].fillna("")))

# COMMAND ----------
# MAGIC %md
# MAGIC ## ✅ Forecasting Complete
# MAGIC
# MAGIC Models registered in UC Model Registry:
# MAGIC - `gov_finance.fiscal_forecaster_totals_revenue@champion`
# MAGIC - `gov_finance.fiscal_forecaster_totals_expenditure@champion`
# MAGIC
# MAGIC Proceed to **04_anomaly_detection**
