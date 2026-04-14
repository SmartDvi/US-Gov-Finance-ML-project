"""
gov_finance_ml/src/data/ingestion.py
Bronze-layer ingestion: raw CSV → Delta Lake.
Supports batch load (initial) and incremental append.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType,
)

from src.utils.spark_utils import add_audit_columns, write_delta

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Raw column rename map (original → snake_case)
# ─────────────────────────────────────────────

def _sanitize_column_name(name: str) -> str:
    """Convert arbitrary header to safe snake_case identifier."""
    name = name.strip().lower()
    name = re.sub(r"[\s.]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────

RAW_SCHEMA = StructType([
    StructField("state", StringType(), True),
    StructField("year", IntegerType(), True),
    StructField("totals_capital_outlay", DoubleType(), True),
    StructField("totals_revenue", DoubleType(), True),
    StructField("totals_expenditure", DoubleType(), True),
    StructField("totals_general_expenditure", DoubleType(), True),
    StructField("totals_general_revenue", DoubleType(), True),
    StructField("totals_insurance_trust_revenue", DoubleType(), True),
    StructField("totals_intergovernmental", DoubleType(), True),
    StructField("totals_license_tax", DoubleType(), True),
    StructField("totals_debt_at_end_of_fiscal_year", DoubleType(), True),
    StructField("details_welfare_welfare_institution_total_expenditure", DoubleType(), True),
    StructField("details_natural_resources_parks_parks_total_expenditure", DoubleType(), True),
    StructField("details_transportation_highways_highways_total_expenditure", DoubleType(), True),
    StructField("details_insurance_benefits_and_repayments", DoubleType(), True),
    StructField("details_interest_on_debt", DoubleType(), True),
    StructField("details_interest_on_general_debt", DoubleType(), True),
    StructField("details_miscellaneous_general_revenue", DoubleType(), True),
    StructField("details_other_taxes", DoubleType(), True),
    StructField("details_police_protection", DoubleType(), True),
])


# ─────────────────────────────────────────────
# Ingestion functions
# ─────────────────────────────────────────────

class BronzeIngestion:
    """
    Handles raw data → Bronze Delta table.

    Parameters
    ----------
    spark : SparkSession
    config : dict
        Full project config dict (from config/config.yaml).
    """

    def __init__(self, spark: SparkSession, config: dict) -> None:
        self.spark = spark
        self.cfg = config
        self.target_table = config["data"]["raw_table"]

    # ── public API ──────────────────────────────

    def ingest_csv(
        self,
        path: str,
        mode: str = "overwrite",
        header: bool = True,
        delimiter: str = ",",
    ) -> DataFrame:
        """
        Read raw CSV from DBFS / Unity Catalog volume and land into Bronze.

        Parameters
        ----------
        path    : DBFS path or UC volume path, e.g. 'dbfs:/mnt/raw/state_finances.csv'
        mode    : 'overwrite' for initial load, 'append' for incremental
        """
        logger.info("Reading CSV from %s", path)
        raw_df = (
            self.spark.read
            .option("header", str(header).lower())
            .option("delimiter", delimiter)
            .option("inferSchema", "false")   # always use explicit schema
            .option("nanValue", "")
            .option("nullValue", "")
            .csv(path)
        )

        logger.info(
            "Raw CSV loaded: %d rows × %d cols", raw_df.count(), len(raw_df.columns)
        )

        cleaned = self._rename_and_cast(raw_df)
        audited = add_audit_columns(cleaned)
        self._validate_bronze(audited)

        write_delta(
            audited,
            self.target_table,
            mode=mode,
            partition_by=["state"],
        )
        logger.info("Bronze ingestion complete → %s", self.target_table)
        return audited

    def ingest_dataframe(self, df: DataFrame, mode: str = "overwrite") -> DataFrame:
        """Accept a pre-built Spark DataFrame (e.g. from a Databricks Auto Loader stream)."""
        cleaned = self._rename_and_cast(df)
        audited = add_audit_columns(cleaned)
        self._validate_bronze(audited)
        write_delta(audited, self.target_table, mode=mode, partition_by=["state"])
        return audited

    def auto_loader_stream(self, cloud_path: str, checkpoint: str) -> None:
        """
        Structured Streaming via Databricks Auto Loader (cloudFiles).
        Useful for continuous ingestion from S3 / ADLS.
        """
        logger.info("Starting Auto Loader stream from %s", cloud_path)
        stream_df = (
            self.spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "csv")
            .option("cloudFiles.schemaLocation", f"{checkpoint}/schema")
            .option("header", "true")
            .load(cloud_path)
        )
        cleaned = self._rename_and_cast(stream_df)
        (
            cleaned.writeStream
            .format("delta")
            .outputMode("append")
            .option("checkpointLocation", f"{checkpoint}/bronze")
            .trigger(availableNow=True)          # process-once trigger for scheduled jobs
            .toTable(self.target_table)
        )

    # ── private helpers ──────────────────────────

    def _rename_and_cast(self, df: DataFrame) -> DataFrame:
        """Rename columns to snake_case and cast to correct types."""
        # Build rename map based on sanitized names
        rename_map = {c: _sanitize_column_name(c) for c in df.columns}
        for old, new in rename_map.items():
            if old != new:
                df = df.withColumnRenamed(old, new)

        # Cast numeric columns
        numeric_cols = [
            f.name for f in RAW_SCHEMA.fields
            if isinstance(f.dataType, (DoubleType, IntegerType))
        ]
        for col in numeric_cols:
            if col in df.columns:
                df = df.withColumn(col, F.col(col).cast(DoubleType()))

        # Year as integer
        if "year" in df.columns:
            df = df.withColumn("year", F.col("year").cast(IntegerType()))

        # Trim state strings
        if "state" in df.columns:
            df = df.withColumn("state", F.upper(F.trim(F.col("state"))))

        return df

    def _validate_bronze(self, df: DataFrame) -> None:
        """Fail fast on critical data quality issues."""
        required = ["state", "year", "totals_revenue", "totals_expenditure"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Bronze validation failed — missing columns: {missing}")

        null_state = df.filter(F.col("state").isNull()).count()
        null_year = df.filter(F.col("year").isNull()).count()
        if null_state > 0 or null_year > 0:
            raise ValueError(
                f"Bronze validation failed — null state={null_state}, null year={null_year}"
            )
        logger.info("Bronze validation passed ✓")
