"""
Stage 1 — Ingestion: raw CSV -> clean, validated Spark DataFrame.

The raw file (US Census "State Government Finances", via CORGIS) has one row
per (state, year) and ~30 dollar columns (thousands of USD) with headers
such as "Totals.Capital outlay" or "Details.Welfare.Welfare Institution
Total Expenditure". This stage:

1. Renames headers to snake_case (totals_capital_outlay, ...)
2. Casts year -> int and every money column -> double
3. Removes rows that are not a single state (e.g. "UNITED STATES")
4. Validates the data and fails fast if something is wrong
"""

from __future__ import annotations

import logging
import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

ID_COLS = ["state", "year"]

# Columns the downstream feature engineering relies on.
REQUIRED_COLS = ID_COLS + [
    "totals_revenue",
    "totals_expenditure",
    "totals_general_revenue",
    "totals_general_expenditure",
    "totals_capital_outlay",
    "totals_debt_at_end_of_fiscal_year",
    "totals_intergovernmental",
    "totals_tax",
    "totals_insurance_trust_revenue",
    "details_welfare_welfare_institution_total_expenditure",
    "details_education_education_total",
    "details_health_health_total_expenditure",
    "details_transportation_highways_highways_total_expenditure",
    "details_police_protection",
    "details_correction_correction_total",
    "details_interest_on_general_debt",
    "details_miscellaneous_general_revenue",
]


def sanitize_column_name(name: str) -> str:
    """'Totals. Debt at end of fiscal year' -> 'totals_debt_at_end_of_fiscal_year'."""
    name = name.strip().lower()
    name = re.sub(r"[\s.]+", "_", name)       # spaces and dots -> underscore
    name = re.sub(r"[^a-z0-9_]", "", name)    # drop any other symbol
    return re.sub(r"_+", "_", name).strip("_")


def load_raw_csv(spark: SparkSession, path: str) -> DataFrame:
    """Read the CSV with every column as string; types are set explicitly in clean()."""
    logger.info("Reading raw CSV: %s", path)
    return spark.read.option("header", True).option("inferSchema", False).csv(path)


def clean(df: DataFrame, exclude_states: list[str]) -> DataFrame:
    """Rename columns, cast types and drop non-state rows."""
    # toDF renames all columns at once (safer than withColumnRenamed for names with dots)
    df = df.toDF(*[sanitize_column_name(c) for c in df.columns])

    money_cols = [c for c in df.columns if c not in ID_COLS]
    df = df.select(
        F.upper(F.trim(F.col("state"))).alias("state"),
        F.col("year").cast("int").alias("year"),
        *[F.col(c).cast("double").alias(c) for c in money_cols],
    )

    excluded = [s.upper() for s in exclude_states]
    return df.filter(~F.col("state").isin(excluded))


def validate(df: DataFrame) -> None:
    """Raise ValueError on problems that would silently corrupt the models."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    null_ids = df.filter(F.col("state").isNull() | F.col("year").isNull()).count()
    if null_ids:
        raise ValueError(f"{null_ids} rows have a null state or year")

    duplicates = df.groupBy(ID_COLS).count().filter(F.col("count") > 1).count()
    if duplicates:
        raise ValueError(f"{duplicates} duplicated (state, year) pairs")

    # Nulls in money columns are not fatal (ratios become null and rows are
    # skipped by the models), but they are worth knowing about.
    null_counts = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in REQUIRED_COLS]
    ).first().asDict()
    columns_with_nulls = {c: n for c, n in null_counts.items() if n}
    if columns_with_nulls:
        logger.warning("Null values found: %s", columns_with_nulls)

    logger.info("Validation passed")


def run_ingestion(spark: SparkSession, csv_path: str, exclude_states: list[str]) -> DataFrame:
    df = clean(load_raw_csv(spark, csv_path), exclude_states)
    validate(df)
    stats = df.agg(
        F.count("*").alias("rows"),
        F.countDistinct("state").alias("states"),
        F.min("year").alias("min_year"),
        F.max("year").alias("max_year"),
    ).first()
    logger.info(
        "Ingested %d rows | %d states | years %d-%d",
        stats["rows"], stats["states"], stats["min_year"], stats["max_year"],
    )
    return df
