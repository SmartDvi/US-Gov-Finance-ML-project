# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Multi-Objective Resource Optimization
# MAGIC **Model D: Genetic Algorithm — Pareto-Optimal Budget Allocation**
# MAGIC
# MAGIC Finds the optimal split of general revenue across:
# MAGIC - Welfare
# MAGIC - Highways
# MAGIC - Police
# MAGIC - Parks
# MAGIC - Debt Service
# MAGIC - Other
# MAGIC
# MAGIC Subject to: min/max share constraints per category.
# MAGIC Objective: minimise debt growth + maximise welfare + infrastructure outcomes.

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Repos/gov_finance_ml")

from src.utils.spark_utils import get_spark, load_config, read_delta
from src.models.resource_optimizer import ResourceOptimizer, BUDGET_CATEGORIES
from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt

spark = get_spark()
cfg = load_config()

# COMMAND ----------
# MAGIC %md ## 6.1 — Run Optimizer

# COMMAND ----------

optimizer = ResourceOptimizer(spark, cfg)
opt_df = optimizer.run()

display(opt_df.orderBy(F.col("fitness_score").desc()))

# COMMAND ----------
# MAGIC %md ## 6.2 — Actual vs Optimal Comparison

# COMMAND ----------

gold_df = read_delta(spark, cfg["data"]["gold_table"])
comparison = optimizer.compare_actual_vs_optimal(gold_df)
print("Top states where WELFARE is under-allocated (negative gap = under-spend):")
print(comparison.sort_values("welfare_gap").head(10).to_string(index=False))
print("\nTop states where HIGHWAYS need more investment:")
print(comparison.sort_values("highway_gap").head(10).to_string(index=False))
display(spark.createDataFrame(comparison))

# COMMAND ----------
# MAGIC %md ## 6.3 — Budget Allocation Visualisation

# COMMAND ----------

opt_pdf = opt_df.limit(10).toPandas()
share_cols = [f"optimal_{cat}_share" for cat in BUDGET_CATEGORIES]

fig, ax = plt.subplots(figsize=(14, 6))
bottom = pd.Series([0.0] * len(opt_pdf))
colors = ["#2196F3", "#4CAF50", "#F44336", "#9C27B0", "#FF9800", "#607D8B"]

for cat, col, color in zip(BUDGET_CATEGORIES, share_cols, colors):
    if col in opt_pdf.columns:
        bars = opt_pdf[col].values
        ax.bar(opt_pdf["state"], bars, bottom=bottom, label=cat.replace("_", " ").title(), color=color)
        bottom += pd.Series(bars)

ax.set_xlabel("State")
ax.set_ylabel("Budget Share")
ax.set_title("Optimal Budget Allocation by State (GA Optimised)")
ax.legend(loc="upper right", fontsize=8)
ax.set_xticklabels(opt_pdf["state"], rotation=45, ha="right")
plt.tight_layout()
display(fig)

# COMMAND ----------
# MAGIC %md ## 6.4 — Sensitivity Analysis (Welfare Weight)

# COMMAND ----------

import numpy as np
sensitivity_results = []
for welfare_weight in np.arange(0.1, 0.6, 0.1):
    cfg_copy = cfg.copy()
    cfg_copy["models"] = dict(cfg["models"])
    cfg_copy["models"]["optimizer"] = dict(cfg["models"]["optimizer"])
    cfg_copy["models"]["optimizer"]["objective_weights"] = dict(cfg["models"]["optimizer"]["objective_weights"])
    cfg_copy["models"]["optimizer"]["objective_weights"]["maximize_welfare"] = round(float(welfare_weight), 2)

    test_opt = ResourceOptimizer(spark, cfg_copy)
    sample_budget = 10_000_000_000
    context = {"current_debt": 5e9, "total_revenue": sample_budget, "fiscal_health_index": 0.6}
    best_alloc, _ = test_opt._run_ga(sample_budget, context)
    sensitivity_results.append({
        "welfare_weight": round(float(welfare_weight), 2),
        "optimal_welfare_share": round(float(best_alloc[0]), 4),
        "optimal_highway_share": round(float(best_alloc[1]), 4),
        "optimal_police_share": round(float(best_alloc[2]), 4),
    })

display(spark.createDataFrame(pd.DataFrame(sensitivity_results)))

# COMMAND ----------
# MAGIC %md ## 6.5 — Policy Recommendation Report

# COMMAND ----------

print("=" * 60)
print("  GOVERNMENT FINANCE ML — POLICY RECOMMENDATIONS")
print("=" * 60)

top_states = opt_df.orderBy(F.col("fitness_score").desc()).limit(5).toPandas()
for _, row in top_states.iterrows():
    print(f"\nState: {row['state']}")
    print(f"  Total Budget   : ${row['total_budget']:>15,.0f}")
    print(f"  Welfare        : {row['optimal_welfare_share']*100:.1f}%  (${row['optimal_welfare_amount']:,.0f})")
    print(f"  Highways       : {row['optimal_highways_share']*100:.1f}%  (${row['optimal_highways_amount']:,.0f})")
    print(f"  Police         : {row['optimal_police_share']*100:.1f}%  (${row['optimal_police_amount']:,.0f})")
    print(f"  Parks          : {row['optimal_parks_share']*100:.1f}%  (${row['optimal_parks_amount']:,.0f})")
    print(f"  Debt Service   : {row['optimal_debt_service_share']*100:.1f}%")
    print(f"  Fitness Score  : {row['fitness_score']:.4f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## ✅ Resource Optimization Complete
# MAGIC
# MAGIC Output table: `gov_finance.gold.state_finances_features_optimal_allocations`
# MAGIC
# MAGIC ### Pipeline Summary
# MAGIC
# MAGIC | Model | Purpose | Output |
# MAGIC |-------|---------|--------|
# MAGIC | **Fiscal Forecaster** | 5-year revenue/expenditure forecast | Early warning signals |
# MAGIC | **Anomaly Detector** | Flag fraud/waste in spending | Audit priority list |
# MAGIC | **State Clustering** | Group states by financial DNA | Peer benchmarking |
# MAGIC | **Resource Optimizer** | Pareto-optimal budget allocation | Policy recommendations |
