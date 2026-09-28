"""Shared pytest fixtures: a small local SparkSession and synthetic finance data."""

from __future__ import annotations

import copy
import random

import pytest
from pyspark.sql import SparkSession

from src.data.ingestion import REQUIRED_COLS
from src.utils.spark_utils import load_config

STATES = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF", "HOTEL"]
# Same gap as the real data: 2005-2011 is missing
YEARS = list(range(1992, 2005)) + list(range(2012, 2020))


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("gov_finance_tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "1g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture
def config():
    """The real config with smaller settings so tests run fast."""
    cfg = copy.deepcopy(load_config())
    cfg["forecaster"]["param_grid"] = {"max_depth": [2], "num_trees": [10]}
    cfg["clustering"]["k_range"] = [2, 3]
    return cfg


def make_clean_rows(seed: int = 7) -> list[dict]:
    """Synthetic data shaped like the output of ingestion.clean()."""
    rng = random.Random(seed)
    money_cols = [c for c in REQUIRED_COLS if c not in ("state", "year")] + [
        "details_natural_resources_parks_parks_total_expenditure"
    ]
    rows = []
    for i, state in enumerate(STATES):
        size = 1e6 * (i + 1)           # states differ in size
        for year in YEARS:
            size *= 1 + rng.gauss(0.04, 0.03)
            row = {"state": state, "year": year}
            for col in money_cols:
                row[col] = size * rng.uniform(0.05, 0.3)
            row["totals_revenue"] = size
            row["totals_general_revenue"] = size * 0.8
            row["totals_expenditure"] = size * rng.uniform(0.9, 1.05)
            row["totals_general_expenditure"] = size * 0.7
            rows.append(row)
    return rows


@pytest.fixture
def clean_df(spark):
    return spark.createDataFrame(make_clean_rows())
