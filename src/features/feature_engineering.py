"""
gov_finance_ml/src/features/feature_engineering.py
Silver → Gold feature engineering.

Produces:
- Lag features
- Rolling mean / std features
- Financial ratio features (policy-insight driven)
- Year-over-year growth rates
- Per-capita normalisation placeholders
- PySpark ML-ready output (VectorAssembler-compatible)
"""

from __future__ import annotations

import logging
from typing import Optional

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer, MinMaxScaler, StandardScaler, VectorAssembler,
)
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.utils.spark_utils import read_delta, write_delta

logger = logging.getLogger(__name__)

# Small constant to avoid division by zero
_EPS = 1.0


class FeatureEngineer:
    """
    Builds the Gold analytical feature layer from Silver data.

    All features are documented inline with the business insight they encode.
    """

    def __init__(self, spark: SparkSession, config: dict) -> None:
        self.spark = spark
        self.cfg = config
        self.source_table = config["data"]["silver_table"]
        self.target_table = config["data"]["gold_table"]
        self.lag_windows = config["feature_engineering"]["lag_windows"]
        self.roll_windows = config["feature_engineering"]["rolling_windows"]

    # ── Public API ────────────────────────────────

    def run(self, input_df: Optional[DataFrame] = None) -> DataFrame:
        df = input_df if input_df is not None else read_delta(self.spark, self.source_table)
        logger.info("Feature engineering: %d input rows", df.count())

        df = self._add_ratio_features(df)
        df = self._add_growth_rates(df)
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_fiscal_health_index(df)
        df = self._add_social_investment_score(df)
        df = self._add_debt_sustainability_flags(df)

        write_delta(df, self.target_table, mode="overwrite", partition_by=["state"])
        logger.info("Gold feature layer written → %s", self.target_table)
        return df

    def build_ml_pipeline(
        self, feature_cols: list[str], scale: str = "standard"
    ) -> Pipeline:
        """
        Return a fitted-ready PySpark ML Pipeline:
        Imputer → VectorAssembler → Scaler

        Parameters
        ----------
        feature_cols : list of column names to include
        scale        : 'standard' (zero-mean unit-var) or 'minmax' ([0,1])
        """
        imputer = Imputer(
            inputCols=feature_cols,
            outputCols=[f"{c}_imp" for c in feature_cols],
            strategy="median",
        )
        assembler = VectorAssembler(
            inputCols=[f"{c}_imp" for c in feature_cols],
            outputCol="features_raw",
            handleInvalid="skip",
        )
        if scale == "minmax":
            scaler = MinMaxScaler(inputCol="features_raw", outputCol="features")
        else:
            scaler = StandardScaler(
                inputCol="features_raw", outputCol="features",
                withMean=True, withStd=True,
            )
        return Pipeline(stages=[imputer, assembler, scaler])

    # ── Feature groups ────────────────────────────

    def _add_ratio_features(self, df: DataFrame) -> DataFrame:
        """
        Insight: fiscal stress, social safety-net efficiency, infrastructure balance.
        """
        cfg_pairs = self.cfg["feature_engineering"]["ratio_pairs"]

        # -- Fiscal stress --
        df = df.withColumn(
            "expenditure_to_revenue_ratio",
            F.col("totals_expenditure") / (F.col("totals_revenue") + _EPS),
        )
        df = df.withColumn(
            "debt_to_revenue_ratio",
            F.col("totals_debt_at_end_of_fiscal_year") / (F.col("totals_revenue") + _EPS),
        )

        # -- Social safety net efficiency --
        df = df.withColumn(
            "welfare_to_insurance_ratio",
            F.col("details_welfare_welfare_institution_total_expenditure")
            / (F.col("details_insurance_benefits_and_repayments") + _EPS),
        )

        # -- Infrastructure spend share --
        df = df.withColumn(
            "highway_expenditure_share",
            F.col("details_transportation_highways_highways_total_expenditure")
            / (F.col("totals_general_expenditure") + _EPS),
        )
        df = df.withColumn(
            "capital_outlay_share",
            F.col("totals_capital_outlay") / (F.col("totals_expenditure") + _EPS),
        )

        # -- Welfare share --
        df = df.withColumn(
            "welfare_expenditure_share",
            F.col("details_welfare_welfare_institution_total_expenditure")
            / (F.col("totals_general_expenditure") + _EPS),
        )

        # -- Police vs parks balance --
        df = df.withColumn(
            "police_vs_parks_ratio",
            F.col("details_police_protection")
            / (F.col("details_natural_resources_parks_parks_total_expenditure") + _EPS),
        )

        # -- Intergovernmental dependency --
        df = df.withColumn(
            "intergovernmental_dependency",
            F.col("totals_intergovernmental") / (F.col("totals_general_revenue") + _EPS),
        )

        # -- Interest burden --
        df = df.withColumn(
            "interest_burden",
            F.col("details_interest_on_general_debt") / (F.col("totals_revenue") + _EPS),
        )

        # -- Misc revenue opacity (anomaly proxy) --
        df = df.withColumn(
            "misc_revenue_share",
            F.col("details_miscellaneous_general_revenue") / (F.col("totals_general_revenue") + _EPS),
        )

        logger.info("Ratio features added")
        return df

    def _add_growth_rates(self, df: DataFrame) -> DataFrame:
        """Year-over-year % growth for core fiscal indicators."""
        w_state = Window.partitionBy("state").orderBy("year")

        growth_cols = [
            "totals_revenue", "totals_expenditure",
            "totals_debt_at_end_of_fiscal_year",
            "details_transportation_highways_highways_total_expenditure",
            "details_welfare_welfare_institution_total_expenditure",
        ]
        for col in growth_cols:
            if col not in df.columns:
                continue
            prev_col = F.lag(F.col(col), 1).over(w_state)
            df = df.withColumn(
                f"{col}_yoy_growth",
                (F.col(col) - prev_col) / (prev_col + _EPS),
            )
        logger.info("YoY growth rates added")
        return df

    def _add_lag_features(self, df: DataFrame) -> DataFrame:
        """Lagged revenue and expenditure for time-series models."""
        w_state = Window.partitionBy("state").orderBy("year")
        lag_cols = ["totals_revenue", "totals_expenditure", "debt_to_revenue_ratio"]

        for col in lag_cols:
            if col not in df.columns:
                continue
            for lag in self.lag_windows:
                df = df.withColumn(
                    f"{col}_lag{lag}",
                    F.lag(F.col(col), lag).over(w_state),
                )
        logger.info("Lag features added (windows=%s)", self.lag_windows)
        return df

    def _add_rolling_features(self, df: DataFrame) -> DataFrame:
        """Rolling mean and std over multiple windows."""
        w_state = Window.partitionBy("state").orderBy("year")
        roll_cols = ["totals_revenue", "totals_expenditure", "totals_capital_outlay"]

        for col in roll_cols:
            if col not in df.columns:
                continue
            for win in self.roll_windows:
                w = w_state.rowsBetween(-(win - 1), 0)
                df = df.withColumn(f"{col}_roll{win}_mean", F.avg(F.col(col)).over(w))
                df = df.withColumn(f"{col}_roll{win}_std", F.stddev(F.col(col)).over(w))
        logger.info("Rolling features added (windows=%s)", self.roll_windows)
        return df

    def _add_fiscal_health_index(self, df: DataFrame) -> DataFrame:
        """
        Composite Fiscal Health Index (FHI) ∈ [0, 1].
        Higher = healthier.
        Penalises: high debt/revenue, high expenditure/revenue.
        Rewards: positive revenue growth.
        """
        rev_growth = F.coalesce(F.col("totals_revenue_yoy_growth"), F.lit(0.0))
        debt_stress = F.least(F.col("debt_to_revenue_ratio") / 2.0, F.lit(1.0))
        exp_stress = F.least(F.col("expenditure_to_revenue_ratio") - 1.0, F.lit(1.0))
        exp_stress = F.greatest(exp_stress, F.lit(0.0))

        fhi = F.greatest(
            F.lit(0.0),
            F.least(
                F.lit(1.0),
                F.lit(0.5)
                + F.lit(0.25) * rev_growth
                - F.lit(0.25) * debt_stress
                - F.lit(0.25) * exp_stress,
            ),
        )
        return df.withColumn("fiscal_health_index", fhi)

    def _add_social_investment_score(self, df: DataFrame) -> DataFrame:
        """
        Social Investment Score (SIS).
        Rewards welfare + parks spend relative to total expenditure.
        Penalises imbalance (all police, no parks).
        """
        welfare_share = F.coalesce(F.col("welfare_expenditure_share"), F.lit(0.0))
        parks_share = (
            F.col("details_natural_resources_parks_parks_total_expenditure")
            / (F.col("totals_general_expenditure") + _EPS)
        )
        police_share = (
            F.col("details_police_protection")
            / (F.col("totals_general_expenditure") + _EPS)
        )

        sis = F.least(F.lit(1.0), welfare_share * 3.0 + parks_share * 2.0) - (police_share * 0.5)
        sis = F.greatest(sis, F.lit(0.0))
        return df.withColumn("social_investment_score", sis)

    def _add_debt_sustainability_flags(self, df: DataFrame) -> DataFrame:
        """
        Binary flags for policy rules.
        - debt_critical: debt > 2× revenue
        - expenditure_over_revenue: state spending beyond means
        - high_interest_burden: interest on debt > 5% of revenue
        """
        df = df.withColumn(
            "debt_critical",
            (F.col("debt_to_revenue_ratio") > 2.0).cast("int"),
        )
        df = df.withColumn(
            "expenditure_over_revenue",
            (F.col("expenditure_to_revenue_ratio") > 1.0).cast("int"),
        )
        df = df.withColumn(
            "high_interest_burden",
            (F.col("interest_burden") > 0.05).cast("int"),
        )
        return df

    # ── Feature list getters ───────────────────────

    @property
    def clustering_features(self) -> list[str]:
        return self.cfg["models"]["clustering"]["features"]

    @property
    def anomaly_features(self) -> list[str]:
        return self.cfg["models"]["anomaly_detector"]["features"]
