"""
Monitoring metrics logged to MLflow on every pipeline run.

Three questions a model owner asks after each run:

1. Data quality   — did we receive the data we expected? (rows, states, nulls)
2. Feature drift  — does the newest year look like the data the model was
                    trained on? If not, predictions are less trustworthy.
3. Predictions    — are the forecasts in a sensible range?

Drift uses the standardised mean difference, which is easy to explain:
    drift = |mean(latest year) - mean(earlier years)| / std(earlier years)
A drift of 1.0 means the feature moved by one standard deviation.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def data_quality_metrics(df: DataFrame) -> dict:
    money_cols = [c for c in df.columns if c not in ("state", "year")]
    row = df.agg(
        F.count("*").alias("rows"),
        F.countDistinct("state").alias("states"),
        F.min("year").alias("min_year"),
        F.max("year").alias("max_year"),
        sum(F.sum(F.col(c).isNull().cast("int")) for c in money_cols).alias("null_cells"),
    ).first()
    return row.asDict()


def feature_drift(features_df: DataFrame, feature_cols: list[str]) -> dict[str, float]:
    """Drift of each feature in the latest year versus all earlier years."""
    latest_year = features_df.agg(F.max("year")).first()[0]
    reference = features_df.filter(F.col("year") < latest_year)
    current = features_df.filter(F.col("year") == latest_year)

    ref_stats = reference.agg(
        *[F.avg(c).alias(f"mean_{c}") for c in feature_cols],
        *[F.stddev(c).alias(f"std_{c}") for c in feature_cols],
    ).first()
    cur_stats = current.agg(*[F.avg(c).alias(f"mean_{c}") for c in feature_cols]).first()

    drift = {}
    for c in feature_cols:
        std = ref_stats[f"std_{c}"]
        if std and cur_stats[f"mean_{c}"] is not None:
            drift[c] = round(abs(cur_stats[f"mean_{c}"] - ref_stats[f"mean_{c}"]) / std, 4)
    return drift


def prediction_summary(forecast_df: DataFrame) -> dict:
    row = forecast_df.agg(
        F.avg("predicted_growth").alias("mean_predicted_growth"),
        F.min("predicted_growth").alias("min_predicted_growth"),
        F.max("predicted_growth").alias("max_predicted_growth"),
        F.count("*").alias("states_forecast"),
    ).first()
    return row.asDict()
