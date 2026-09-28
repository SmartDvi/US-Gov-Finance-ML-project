"""
Stage 2 — Feature engineering: clean data -> one feature row per (state, year).

Design rules (important when explaining the project):

* Scale-free features. California's budget is ~100x Wyoming's, so raw dollar
  amounts mostly measure state size. Ratios ("share of spending on welfare")
  and growth rates ("revenue grew 6%") make states comparable.

* Only look backwards. Every feature for year t uses data from year t or
  earlier, so it would be available at the time a forecast is made.

* Respect gaps in the data. The dataset skips 2005-2011. A plain lag() would
  treat 2004 -> 2012 as "one year", producing an 8-year growth labelled as
  a 1-year growth. lag_if_consecutive() returns null in that case.
"""

from __future__ import annotations

import logging

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from src.utils.spark_utils import safe_divide

logger = logging.getLogger(__name__)

def state_window() -> Window:
    """Rows of the same state, ordered by year (used by lag/lead)."""
    return Window.partitionBy("state").orderBy("year")

# Share of general expenditure spent on each function: name -> source column
SPENDING_SHARES = {
    "welfare_share": "details_welfare_welfare_institution_total_expenditure",
    "education_share": "details_education_education_total",
    "health_share": "details_health_health_total_expenditure",
    "highways_share": "details_transportation_highways_highways_total_expenditure",
    "police_share": "details_police_protection",
    "corrections_share": "details_correction_correction_total",
    "parks_share": "details_natural_resources_parks_parks_total_expenditure",
}

# Columns that get a year-over-year growth rate feature: <col>_growth
GROWTH_COLS = [
    "totals_revenue",
    "totals_general_revenue",
    "totals_expenditure",
    "totals_tax",
    "totals_capital_outlay",
    "totals_debt_at_end_of_fiscal_year",
    "details_interest_on_general_debt",
    "details_miscellaneous_general_revenue",
]


def lag_if_consecutive(column: str | Column, offset: int = 1) -> Column:
    """Value from `offset` years earlier, or null if that year is missing in the data."""
    column = F.col(column) if isinstance(column, str) else column
    previous_year = F.lag("year", offset).over(state_window())
    return F.when(
        F.col("year") - previous_year == offset,
        F.lag(column, offset).over(state_window()),
    )


def add_ratio_features(df: DataFrame) -> DataFrame:
    """Budget structure and fiscal stress ratios."""
    general_revenue = F.col("totals_general_revenue")
    general_spend = F.col("totals_general_expenditure")

    df = df.select(
        "*",
        # How is the money spent?
        *[
            safe_divide(F.col(source), general_spend).alias(name)
            for name, source in SPENDING_SHARES.items()
        ],
        safe_divide(F.col("totals_capital_outlay"), F.col("totals_expenditure"))
        .alias("capital_outlay_share"),
        # Where does the money come from?
        safe_divide(F.col("totals_tax"), general_revenue).alias("tax_share_of_general_revenue"),
        safe_divide(F.col("totals_intergovernmental"), general_revenue)
        .alias("intergovernmental_dependency"),
        safe_divide(F.col("details_miscellaneous_general_revenue"), general_revenue)
        .alias("misc_revenue_share"),
        safe_divide(F.col("totals_insurance_trust_revenue"), F.col("totals_revenue"))
        .alias("insurance_trust_share_of_revenue"),
        # Fiscal stress
        safe_divide(F.col("totals_expenditure"), F.col("totals_revenue"))
        .alias("expenditure_to_revenue_ratio"),
        safe_divide(F.col("totals_debt_at_end_of_fiscal_year"), general_revenue)
        .alias("debt_to_revenue_ratio"),
        safe_divide(F.col("details_interest_on_general_debt"), general_revenue)
        .alias("interest_burden"),
    )
    return df.withColumn(
        "is_deficit", (F.col("totals_expenditure") > F.col("totals_revenue")).cast("int")
    )


def add_growth_features(df: DataFrame) -> DataFrame:
    """Year-over-year growth, e.g. 0.05 = +5% versus the previous year."""
    growth = []
    for col in GROWTH_COLS:
        previous = lag_if_consecutive(col)
        # Divide by |previous| so a change from -10 to -5 is +50%, not -50%
        growth.append(safe_divide(F.col(col) - previous, F.abs(previous)).alias(f"{col}_growth"))
    df = df.select("*", *growth)

    # Momentum: last year's revenue growth
    df = df.withColumn(
        "totals_revenue_growth_prev", lag_if_consecutive("totals_revenue_growth")
    )

    # Macro signal: average revenue growth across all states in the same year
    # (captures recessions / booms that hit every state at once).
    return df.withColumn(
        "national_revenue_growth",
        F.avg("totals_revenue_growth").over(Window.partitionBy("year")),
    )


def build_features(clean_df: DataFrame) -> DataFrame:
    df = add_ratio_features(clean_df)
    df = add_growth_features(df)
    logger.info("Feature table built with %d columns", len(df.columns))
    return df
