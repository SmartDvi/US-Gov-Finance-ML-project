"""
Model C — Anomaly detection on year-over-year changes (robust z-score).

Business question: "Which state-years had unusual swings in revenue,
spending, borrowing or miscellaneous income that an auditor should review?"

How it works
------------
* Features are year-over-year growth rates (scale-free, so a small and a
  large state can be compared).
* For each feature we compute the median and the MAD (median absolute
  deviation) over all state-years. The "modified z-score"
        z = 0.6745 * (value - median) / MAD
  measures how unusual a value is. Median/MAD are used instead of mean/std
  because the outliers we are hunting would inflate the mean and std and
  hide themselves.
* anomaly_score = the largest |z| across features. A row is an anomaly when
  it exceeds the threshold (see config.yaml for why 5.0 rather than the
  textbook 3.5 from Iglewicz & Hoaglin). `reasons` lists every feature above the threshold, so each flag
  is explainable.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

MAD_TO_STD = 0.6745  # makes the MAD comparable to a standard deviation for normal data


class AnomalyDetector:
    def __init__(self, config: dict) -> None:
        cfg = config["anomaly"]
        self.feature_cols = cfg["features"]
        self.threshold = cfg["z_threshold"]
        self.medians: dict[str, float] = {}
        self.mads: dict[str, float] = {}

    def fit(self, df: DataFrame) -> "AnomalyDetector":
        """Learn the median and MAD of each feature."""
        self.medians = df.agg(
            *[F.percentile_approx(c, 0.5).alias(c) for c in self.feature_cols]
        ).first().asDict()
        self.mads = df.agg(
            *[
                F.percentile_approx(F.abs(F.col(c) - F.lit(self.medians[c])), 0.5).alias(c)
                for c in self.feature_cols
            ]
        ).first().asDict()
        logger.info("Anomaly detector fitted on %d features", len(self.feature_cols))
        return self

    def score(self, df: DataFrame) -> DataFrame:
        """Add z-scores, anomaly_score, is_anomaly and reasons to every scorable row."""
        if not self.medians:
            raise RuntimeError("Call fit() before score()")

        z_cols = []
        for c in self.feature_cols:
            mad = self.mads[c] or 1e-9  # guard against a constant feature
            z_cols.append((F.lit(MAD_TO_STD) * (F.col(c) - self.medians[c]) / mad).alias(f"z_{c}"))
        scored = df.select("state", "year", *self.feature_cols, *z_cols)

        abs_z = [F.abs(F.col(f"z_{c}")) for c in self.feature_cols]
        scored = scored.withColumn("anomaly_score", F.greatest(*abs_z))

        # Human-readable explanation, e.g. "totals_capital_outlay_growth (+5.2)"
        reasons = [
            F.when(
                F.abs(F.col(f"z_{c}")) > self.threshold,
                F.format_string("%s (%+.1f)", F.lit(c), F.col(f"z_{c}")),
            )
            for c in self.feature_cols
        ]
        return (
            scored.filter(F.col("anomaly_score").isNotNull())  # first year after a data gap
            .withColumn("is_anomaly", (F.col("anomaly_score") > self.threshold).cast("int"))
            .withColumn("reasons", F.concat_ws("; ", *reasons))
        )

    def run(self, features_df: DataFrame) -> dict:
        scored = self.fit(features_df).score(features_df).cache()
        n_rows = scored.count()
        n_anomalies = scored.filter(F.col("is_anomaly") == 1).count()
        logger.info("Flagged %d of %d state-years as anomalies", n_anomalies, n_rows)

        return {
            "scores": scored,
            "metrics": {
                "z_threshold": self.threshold,
                "state_years_scored": n_rows,
                "anomalies_flagged": n_anomalies,
                "anomaly_rate_pct": round(100 * n_anomalies / n_rows, 2) if n_rows else 0.0,
                "feature_medians": {k: round(v, 4) for k, v in self.medians.items()},
                "feature_mads": {k: round(v, 4) for k, v in self.mads.items()},
            },
        }


def audit_report(scores: DataFrame) -> DataFrame:
    """Flagged state-years, most unusual first."""
    return (
        scores.filter(F.col("is_anomaly") == 1)
        .select("state", "year", F.round("anomaly_score", 2).alias("anomaly_score"), "reasons")
        .orderBy(F.col("anomaly_score").desc())
    )
