"""
Shared helpers: config loading, SparkSession creation and parquet I/O.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

# src/utils/spark_utils.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML config (defaults to config/config.yaml in the project root)."""
    path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)


def get_spark(config: dict) -> SparkSession:
    """Create (or reuse) a local SparkSession configured from the config file."""
    spark_cfg = config["spark"]
    spark = (
        SparkSession.builder
        .appName(spark_cfg["app_name"])
        .master(spark_cfg["master"])
        .config("spark.driver.memory", spark_cfg["driver_memory"])
        .config("spark.sql.shuffle.partitions", spark_cfg["shuffle_partitions"])
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ── Output paths & I/O ────────────────────────────────────────────

def output_path(config: dict, *parts: str) -> str:
    """Absolute path inside the configured output directory."""
    return str(PROJECT_ROOT / config["paths"]["output_dir"] / Path(*parts))


def write_parquet(df: DataFrame, path: str, partition_by: list[str] | None = None) -> None:
    logger.info("Writing parquet -> %s", path)
    writer = df.write.mode("overwrite")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.parquet(path)


def read_parquet(spark: SparkSession, path: str) -> DataFrame:
    logger.info("Reading parquet <- %s", path)
    return spark.read.parquet(path)


def write_csv_report(df: DataFrame, path: str) -> None:
    """Write a small result table as a single CSV file (folder with one part file)."""
    logger.info("Writing CSV report -> %s", path)
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(path)


def write_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    logger.info("Wrote %s", path)


# ── Column helpers ────────────────────────────────────────────────

def safe_divide(numerator: Column, denominator: Column) -> Column:
    """Division that returns null (not infinity or an error) when the denominator is 0 or null."""
    return F.when(denominator != 0, numerator / denominator)
