"""
gov_finance_ml/src/utils/spark_utils.py
Databricks-aware Spark session factory & helpers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Configuration loader
# ─────────────────────────────────────────────

def load_config(path: Optional[str] = None) -> dict:
    """Load YAML config. Resolves to repo-root config/config.yaml by default."""
    if path is None:
        root = Path(__file__).resolve().parents[3]
        path = root / "config" / "config.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)


# ─────────────────────────────────────────────
# SparkSession factory
# ─────────────────────────────────────────────

def get_spark(app_name: str = "GovFinanceML") -> SparkSession:
    """
    Return the active SparkSession.
    - On Databricks the session already exists; this simply retrieves it.
    - Locally it creates a local session for unit-tests / development.
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.databricks.delta.preview.enabled", "true")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    # Detect Databricks environment
    if _is_databricks():
        logger.info("Databricks runtime detected — reusing active session.")
        return SparkSession.getActiveSession() or builder.getOrCreate()

    # Local / CI fallback
    logger.info("Local environment — creating new SparkSession.")
    return (
        builder
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def _is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


# ─────────────────────────────────────────────
# Delta helpers
# ─────────────────────────────────────────────

def read_delta(spark: SparkSession, table: str) -> DataFrame:
    logger.info("Reading Delta table: %s", table)
    return spark.read.format("delta").table(table)


def write_delta(
    df: DataFrame,
    table: str,
    mode: str = "overwrite",
    partition_by: Optional[list[str]] = None,
    merge_schema: bool = False,
) -> None:
    """Write a DataFrame as a managed Delta table."""
    logger.info("Writing Delta table: %s  (mode=%s)", table, mode)
    writer = df.write.format("delta").mode(mode)
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    if merge_schema:
        writer = writer.option("mergeSchema", "true")
    writer.saveAsTable(table)
    logger.info("Delta write complete → %s", table)


def upsert_delta(
    spark: SparkSession,
    source_df: DataFrame,
    target_table: str,
    merge_keys: list[str],
) -> None:
    """
    Perform a Delta MERGE (upsert) operation.
    Requires Delta Lake on the cluster.
    """
    from delta.tables import DeltaTable  # type: ignore

    key_condition = " AND ".join(
        f"target.{k} = source.{k}" for k in merge_keys
    )

    if DeltaTable.isDeltaTable(spark, target_table):
        delta_tbl = DeltaTable.forName(spark, target_table)
        (
            delta_tbl.alias("target")
            .merge(source_df.alias("source"), key_condition)
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
        logger.info("MERGE complete → %s", target_table)
    else:
        write_delta(source_df, target_table, mode="overwrite")


# ─────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────

def assert_schema(df: DataFrame, expected: StructType, strict: bool = False) -> None:
    """Raise ValueError when required columns are missing."""
    expected_fields = {f.name for f in expected.fields}
    actual_fields = set(df.columns)
    missing = expected_fields - actual_fields
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    if strict:
        extra = actual_fields - expected_fields
        if extra:
            raise ValueError(f"Unexpected columns in DataFrame: {extra}")


# ─────────────────────────────────────────────
# DataFrame quality helpers
# ─────────────────────────────────────────────

def null_report(df: DataFrame) -> DataFrame:
    """Return a single-row DF with null counts per column."""
    spark = df.sparkSession
    null_counts = [
        F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns
    ]
    return df.select(null_counts)


def describe_numeric(df: DataFrame) -> DataFrame:
    numeric_cols = [
        f.name for f in df.schema.fields
        if str(f.dataType) in ("DoubleType", "FloatType", "IntegerType", "LongType")
    ]
    return df.select(numeric_cols).describe()


def add_audit_columns(df: DataFrame) -> DataFrame:
    """Add ingestion timestamp and a monotonically increasing ID."""
    return df.withColumn(
        "_ingested_at", F.current_timestamp()
    ).withColumn("_row_id", F.monotonically_increasing_id())
