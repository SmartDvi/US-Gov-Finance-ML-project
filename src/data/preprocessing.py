"""
gov_finance_ml/src/data/preprocessing.py
Bronze → Silver transformation layer.
Handles cleaning, imputation, outlier clipping & data quality scoring.
"""

from __future__ import annotations

import logging
from typing import Optional

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from src.utils.spark_utils import read_delta, write_delta

logger = logging.getLogger(__name__)


class SilverProcessor:
    """
    Transforms Bronze data into clean Silver Delta table.

    Responsibilities:
    - De-duplicate (state, year) pairs
    - Impute missing financial values using state-level rolling medians
    - Clip extreme outliers using IQR fence
    - Enforce non-negative financial values
    - Compute a per-row Data Quality Score (DQS)
    - Persist to Silver with partition on (state)
    """

    # Columns that should never be negative
    NON_NEGATIVE_COLS = [
        "totals_capital_outlay", "totals_revenue", "totals_expenditure",
        "totals_general_expenditure", "totals_general_revenue",
        "totals_insurance_trust_revenue", "totals_intergovernmental",
        "totals_license_tax", "totals_debt_at_end_of_fiscal_year",
        "details_welfare_welfare_institution_total_expenditure",
        "details_natural_resources_parks_parks_total_expenditure",
        "details_transportation_highways_highways_total_expenditure",
        "details_insurance_benefits_and_repayments",
        "details_interest_on_debt", "details_interest_on_general_debt",
        "details_police_protection",
    ]

    NUMERIC_COLS = NON_NEGATIVE_COLS + [
        "details_miscellaneous_general_revenue",
        "details_other_taxes",
    ]

    def __init__(self, spark: SparkSession, config: dict) -> None:
        self.spark = spark
        self.cfg = config
        self.source_table = config["data"]["raw_table"]
        self.target_table = config["data"]["silver_table"]

    # ── Public API ────────────────────────────────

    def run(self, input_df: Optional[DataFrame] = None) -> DataFrame:
        """Execute full Bronze → Silver pipeline."""
        df = input_df if input_df is not None else read_delta(self.spark, self.source_table)
        logger.info("Silver processing: %d raw rows", df.count())

        df = self._deduplicate(df)
        df = self._enforce_year_range(df)
        df = self._clip_outliers(df)
        df = self._impute_missing(df)
        df = self._enforce_non_negative(df)
        df = self._add_dqs(df)
        df = self._add_split_column(df)

        write_delta(df, self.target_table, mode="overwrite", partition_by=["state"])
        logger.info("Silver write complete → %s  (%d rows)", self.target_table, df.count())
        return df

    # ── Steps ─────────────────────────────────────

    def _deduplicate(self, df: DataFrame) -> DataFrame:
        """Keep the most-recently-ingested record per (state, year)."""
        w = Window.partitionBy("state", "year").orderBy(F.col("_ingested_at").desc())
        return (
            df.withColumn("_rn", F.row_number().over(w))
            .filter(F.col("_rn") == 1)
            .drop("_rn")
        )

    def _enforce_year_range(self, df: DataFrame) -> DataFrame:
        min_yr = self.cfg["data"]["min_year"]
        max_yr = self.cfg["data"]["max_year"]
        before = df.count()
        df = df.filter(F.col("year").between(min_yr, max_yr))
        after = df.count()
        logger.info("Year range filter [%d–%d]: dropped %d rows", min_yr, max_yr, before - after)
        return df

    def _clip_outliers(self, df: DataFrame) -> DataFrame:
        """
        Clip values beyond [Q1 - 3*IQR, Q3 + 3*IQR] per numeric column.
        Uses a global (not per-state) fence to catch extreme data errors.
        """
        for col in self.NUMERIC_COLS:
            if col not in df.columns:
                continue
            quantiles = df.approxQuantile(col, [0.25, 0.75], 0.01)
            if len(quantiles) < 2:
                continue
            q1, q3 = quantiles
            iqr = q3 - q1
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            df = df.withColumn(col, F.least(F.greatest(F.col(col), F.lit(lower)), F.lit(upper)))
        logger.info("Outlier clipping applied to %d columns", len(self.NUMERIC_COLS))
        return df

    def _impute_missing(self, df: DataFrame) -> DataFrame:
        """
        State-level rolling median imputation using a 3-year backward window.
        Falls back to global median if state-level median is null.
        """
        w_state = (
            Window.partitionBy("state")
            .orderBy("year")
            .rowsBetween(-3, -1)
        )
        for col in self.NUMERIC_COLS:
            if col not in df.columns:
                continue
            rolling_median_col = f"_med_{col}"
            df = df.withColumn(
                rolling_median_col,
                F.percentile_approx(F.col(col), 0.5, 100).over(w_state),
            )

        # Global medians as ultimate fallback
        global_meds = {
            col: df.approxQuantile(col, [0.5], 0.01)[0]
            for col in self.NUMERIC_COLS
            if col in df.columns
        }

        for col in self.NUMERIC_COLS:
            if col not in df.columns:
                continue
            rolling_col = f"_med_{col}"
            global_val = global_meds.get(col, 0.0)
            df = df.withColumn(
                col,
                F.when(F.col(col).isNull(), F.coalesce(F.col(rolling_col), F.lit(global_val)))
                .otherwise(F.col(col))
            ).drop(rolling_col)

        logger.info("Missing value imputation complete")
        return df

    def _enforce_non_negative(self, df: DataFrame) -> DataFrame:
        for col in self.NON_NEGATIVE_COLS:
            if col in df.columns:
                df = df.withColumn(col, F.greatest(F.col(col), F.lit(0.0)))
        return df

    def _add_dqs(self, df: DataFrame) -> DataFrame:
        """
        Data Quality Score ∈ [0, 1].
        Penalises:  null values, zero-revenue rows, debt > 10× revenue.
        """
        total_cols = len(self.NUMERIC_COLS)
        null_penalty = sum(
            F.col(c).isNull().cast("double")
            for c in self.NUMERIC_COLS if c in df.columns
        ) / total_cols

        zero_revenue_penalty = (
            F.when(F.col("totals_revenue") <= 0, F.lit(0.3)).otherwise(F.lit(0.0))
        )
        debt_penalty = F.when(
            (F.col("totals_debt_at_end_of_fiscal_year") / (F.col("totals_revenue") + 1)) > 10,
            F.lit(0.2),
        ).otherwise(F.lit(0.0))

        dqs = F.greatest(
            F.lit(0.0),
            (F.lit(1.0) - null_penalty - zero_revenue_penalty - debt_penalty),
        )
        return df.withColumn("data_quality_score", dqs)

    def _add_split_column(self, df: DataFrame) -> DataFrame:
        """Deterministic train / test split based on year cutoff."""
        cutoff = self.cfg["data"]["train_cutoff_year"]
        return df.withColumn(
            "split",
            F.when(F.col("year") <= cutoff, F.lit("train")).otherwise(F.lit("test")),
        )

    # ── Quality report ────────────────────────────

    def quality_report(self, df: DataFrame) -> dict:
        row_count = df.count()
        null_counts = {
            c: df.filter(F.col(c).isNull()).count()
            for c in self.NUMERIC_COLS if c in df.columns
        }
        avg_dqs = df.select(F.avg("data_quality_score")).collect()[0][0]
        year_range = df.agg(F.min("year"), F.max("year")).collect()[0]
        return {
            "row_count": row_count,
            "null_counts": null_counts,
            "avg_data_quality_score": round(avg_dqs or 0.0, 4),
            "year_range": [year_range[0], year_range[1]],
            "states": df.select("state").distinct().count(),
        }
