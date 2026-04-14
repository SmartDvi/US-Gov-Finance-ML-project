"""
tests/test_pipeline.py
Unit & integration tests for the Government Finance ML pipeline.
Run locally: pytest tests/ -v
Run on Databricks: %run /Workspace/Repos/gov_finance_ml/tests/test_pipeline
"""

from __future__ import annotations

import json
import pytest
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    """Create a local SparkSession for testing."""
    from pyspark.sql import SparkSession
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("gov_finance_test")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


@pytest.fixture(scope="session")
def sample_config():
    """Minimal config dict for testing (avoids I/O)."""
    return {
        "project": {"name": "test", "version": "0.0.1"},
        "data": {
            "raw_table": "test.bronze",
            "silver_table": "test.silver",
            "gold_table": "test.gold",
            "target_revenue_col": "totals_revenue",
            "target_expenditure_col": "totals_expenditure",
            "id_cols": ["state", "year"],
            "min_year": 1992,
            "max_year": 2019,
            "train_cutoff_year": 2014,
        },
        "feature_engineering": {
            "lag_windows": [1, 2],
            "rolling_windows": [2, 3],
            "ratio_pairs": [
                ["totals_expenditure", "totals_revenue"],
                ["totals_debt_at_end_of_fiscal_year", "totals_revenue"],
            ],
        },
        "models": {
            "fiscal_forecaster": {
                "name": "test_forecaster",
                "horizon_years": 2,
                "prophet": {
                    "changepoint_prior_scale": [0.05],
                    "seasonality_prior_scale": [1.0],
                    "seasonality_mode": ["additive"],
                },
                "xgb": {
                    "n_estimators": [50],
                    "max_depth": [3],
                    "learning_rate": [0.1],
                    "subsample": [0.8],
                },
            },
            "anomaly_detector": {
                "name": "test_anomaly",
                "isolation_forest": {
                    "n_estimators": 50,
                    "contamination": 0.05,
                    "random_state": 42,
                },
                "autoencoder": {
                    "encoding_dim": [8, 4],
                    "epochs": 5,
                    "batch_size": 16,
                    "threshold_percentile": 95,
                },
                "features": [
                    "totals_capital_outlay",
                    "totals_revenue",
                    "totals_expenditure",
                    "details_interest_on_general_debt",
                    "details_miscellaneous_general_revenue",
                    "details_other_taxes",
                ],
            },
            "clustering": {
                "name": "test_clustering",
                "k_range": [2, 3],
                "random_state": 42,
                "features": [
                    "totals_license_tax",
                    "totals_intergovernmental",
                    "details_police_protection",
                    "debt_to_revenue_ratio",
                    "welfare_expenditure_share",
                ],
            },
            "optimizer": {
                "name": "test_optimizer",
                "budget_col": "totals_general_revenue",
                "objective_weights": {
                    "minimize_debt": 0.35,
                    "maximize_welfare": 0.30,
                    "maximize_infrastructure": 0.20,
                    "maximize_public_safety": 0.15,
                },
                "constraint_bounds": {
                    "welfare_min_share": 0.08,
                    "welfare_max_share": 0.30,
                    "highway_min_share": 0.05,
                    "highway_max_share": 0.25,
                    "police_min_share": 0.02,
                    "police_max_share": 0.15,
                },
                "ga": {
                    "population_size": 20,
                    "generations": 10,
                    "crossover_rate": 0.85,
                    "mutation_rate": 0.12,
                },
            },
        },
        "mlflow": {
            "experiment_base": "/test/gov_finance_ml",
            "registered_model_prefix": "test",
            "tracking_uri": "sqlite:///test_mlflow.db",
            "artifact_location": "/tmp/mlflow_test",
        },
    }


@pytest.fixture(scope="session")
def sample_bronze_df(spark):
    """Minimal synthetic Bronze DataFrame."""
    rows = []
    states = ["ALABAMA", "ALASKA", "ARIZONA", "WYOMING", "TEXAS"]
    rng = np.random.default_rng(42)
    for state in states:
        for year in range(1995, 2020):
            base = rng.uniform(1e8, 1e10)
            rows.append({
                "state": state,
                "year": int(year),
                "totals_capital_outlay": float(base * rng.uniform(0.05, 0.15)),
                "totals_revenue": float(base),
                "totals_expenditure": float(base * rng.uniform(0.9, 1.1)),
                "totals_general_expenditure": float(base * rng.uniform(0.7, 0.9)),
                "totals_general_revenue": float(base * rng.uniform(0.8, 1.0)),
                "totals_insurance_trust_revenue": float(base * rng.uniform(0.1, 0.2)),
                "totals_intergovernmental": float(base * rng.uniform(0.2, 0.4)),
                "totals_license_tax": float(base * rng.uniform(0.01, 0.05)),
                "totals_debt_at_end_of_fiscal_year": float(base * rng.uniform(0.5, 2.5)),
                "details_welfare_welfare_institution_total_expenditure": float(base * rng.uniform(0.05, 0.15)),
                "details_natural_resources_parks_parks_total_expenditure": float(base * rng.uniform(0.01, 0.05)),
                "details_transportation_highways_highways_total_expenditure": float(base * rng.uniform(0.05, 0.15)),
                "details_insurance_benefits_and_repayments": float(base * rng.uniform(0.05, 0.15)),
                "details_interest_on_debt": float(base * rng.uniform(0.01, 0.05)),
                "details_interest_on_general_debt": float(base * rng.uniform(0.01, 0.05)),
                "details_miscellaneous_general_revenue": float(base * rng.uniform(0.01, 0.10)),
                "details_other_taxes": float(base * rng.uniform(0.01, 0.05)),
                "details_police_protection": float(base * rng.uniform(0.01, 0.05)),
                "_ingested_at": pd.Timestamp.now(),
                "_row_id": int(rng.integers(1_000_000)),
            })
    return spark.createDataFrame(pd.DataFrame(rows))


# ─────────────────────────────────────────────
# Utility tests
# ─────────────────────────────────────────────

class TestSparkUtils:
    def test_null_report_returns_correct_schema(self, spark, sample_bronze_df):
        from src.utils.spark_utils import null_report
        report = null_report(sample_bronze_df)
        assert "totals_revenue" in report.columns
        assert report.count() == 1

    def test_add_audit_columns(self, spark, sample_bronze_df):
        from src.utils.spark_utils import add_audit_columns
        df = add_audit_columns(sample_bronze_df)
        assert "_ingested_at" in df.columns or "totals_revenue" in df.columns

    def test_sanitize_column_name(self):
        from src.data.ingestion import _sanitize_column_name
        assert _sanitize_column_name("Totals.Revenue") == "totals_revenue"
        assert _sanitize_column_name("Details.Welfare.Welfare Institution Total Expenditure") \
               == "details_welfare_welfare_institution_total_expenditure"
        assert _sanitize_column_name("  Extra  Spaces  ") == "extra__spaces"

    def test_load_config_returns_dict(self):
        from src.utils.spark_utils import load_config
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "data" in cfg
        assert "models" in cfg
        assert "mlflow" in cfg


# ─────────────────────────────────────────────
# Preprocessing tests
# ─────────────────────────────────────────────

class TestSilverProcessor:
    def _get_processor(self, spark, cfg):
        from src.data.preprocessing import SilverProcessor

        class MockProcessor(SilverProcessor):
            def run(self, input_df=None):
                df = input_df
                df = self._deduplicate(df)
                df = self._enforce_year_range(df)
                df = self._clip_outliers(df)
                df = self._impute_missing(df)
                df = self._enforce_non_negative(df)
                df = self._add_dqs(df)
                df = self._add_split_column(df)
                return df

        return MockProcessor(spark, cfg)

    def test_deduplication(self, spark, sample_config, sample_bronze_df):
        proc = self._get_processor(spark, sample_config)
        deduplicated = proc._deduplicate(sample_bronze_df)
        # Should have at most one row per (state, year)
        from pyspark.sql import functions as F
        dupes = (
            deduplicated.groupBy("state", "year")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )
        assert dupes == 0

    def test_year_range_filter(self, spark, sample_config, sample_bronze_df):
        proc = self._get_processor(spark, sample_config)
        filtered = proc._enforce_year_range(sample_bronze_df)
        from pyspark.sql import functions as F
        min_yr = filtered.agg(F.min("year")).collect()[0][0]
        max_yr = filtered.agg(F.max("year")).collect()[0][0]
        assert min_yr >= sample_config["data"]["min_year"]
        assert max_yr <= sample_config["data"]["max_year"]

    def test_non_negative_enforcement(self, spark, sample_config, sample_bronze_df):
        from pyspark.sql import functions as F
        proc = self._get_processor(spark, sample_config)
        result = proc._enforce_non_negative(sample_bronze_df)
        neg_count = result.filter(F.col("totals_revenue") < 0).count()
        assert neg_count == 0

    def test_dqs_range(self, spark, sample_config, sample_bronze_df):
        from pyspark.sql import functions as F
        proc = self._get_processor(spark, sample_config)
        result = proc._add_dqs(sample_bronze_df)
        out_of_range = result.filter(
            (F.col("data_quality_score") < 0) | (F.col("data_quality_score") > 1)
        ).count()
        assert out_of_range == 0

    def test_split_column(self, spark, sample_config, sample_bronze_df):
        from pyspark.sql import functions as F
        proc = self._get_processor(spark, sample_config)
        result = proc._add_split_column(sample_bronze_df)
        assert "split" in result.columns
        invalid_splits = result.filter(
            ~F.col("split").isin(["train", "test"])
        ).count()
        assert invalid_splits == 0

    def test_quality_report(self, spark, sample_config, sample_bronze_df):
        proc = self._get_processor(spark, sample_config)
        silver_df = proc.run(input_df=sample_bronze_df)
        report = proc.quality_report(silver_df)
        assert "row_count" in report
        assert "avg_data_quality_score" in report
        assert 0.0 <= report["avg_data_quality_score"] <= 1.0
        assert report["states"] == 5


# ─────────────────────────────────────────────
# Feature engineering tests
# ─────────────────────────────────────────────

class TestFeatureEngineer:
    def _get_silver_df(self, spark, sample_config, sample_bronze_df):
        from src.data.preprocessing import SilverProcessor

        class MockSilverProcessor(SilverProcessor):
            def run(self, input_df=None):
                df = input_df
                df = self._deduplicate(df)
                df = self._enforce_year_range(df)
                df = self._clip_outliers(df)
                df = self._impute_missing(df)
                df = self._enforce_non_negative(df)
                df = self._add_dqs(df)
                df = self._add_split_column(df)
                return df

        return MockSilverProcessor(spark, sample_config).run(input_df=sample_bronze_df)

    def test_ratio_features_added(self, spark, sample_config, sample_bronze_df):
        from src.features.feature_engineering import FeatureEngineer
        silver_df = self._get_silver_df(spark, sample_config, sample_bronze_df)

        class MockFE(FeatureEngineer):
            def run(self, input_df=None):
                df = input_df
                df = self._add_ratio_features(df)
                df = self._add_growth_rates(df)
                df = self._add_lag_features(df)
                df = self._add_rolling_features(df)
                df = self._add_fiscal_health_index(df)
                df = self._add_social_investment_score(df)
                df = self._add_debt_sustainability_flags(df)
                return df

        fe = MockFE(spark, sample_config)
        gold_df = fe.run(input_df=silver_df)

        expected_features = [
            "debt_to_revenue_ratio", "expenditure_to_revenue_ratio",
            "welfare_expenditure_share", "highway_expenditure_share",
            "fiscal_health_index", "social_investment_score",
            "debt_critical", "expenditure_over_revenue", "high_interest_burden",
        ]
        for feat in expected_features:
            assert feat in gold_df.columns, f"Missing feature: {feat}"

    def test_fiscal_health_index_range(self, spark, sample_config, sample_bronze_df):
        from pyspark.sql import functions as F
        from src.features.feature_engineering import FeatureEngineer

        silver_df = self._get_silver_df(spark, sample_config, sample_bronze_df)

        class MockFE(FeatureEngineer):
            def run(self, input_df=None):
                df = input_df
                df = self._add_ratio_features(df)
                df = self._add_growth_rates(df)
                df = self._add_lag_features(df)
                df = self._add_rolling_features(df)
                df = self._add_fiscal_health_index(df)
                df = self._add_social_investment_score(df)
                df = self._add_debt_sustainability_flags(df)
                return df

        fe = MockFE(spark, sample_config)
        gold_df = fe.run(input_df=silver_df)
        out_of_range = gold_df.filter(
            (F.col("fiscal_health_index") < 0) | (F.col("fiscal_health_index") > 1)
        ).count()
        assert out_of_range == 0

    def test_lag_features_created(self, spark, sample_config, sample_bronze_df):
        from src.features.feature_engineering import FeatureEngineer
        silver_df = self._get_silver_df(spark, sample_config, sample_bronze_df)

        class MockFE(FeatureEngineer):
            def run(self, input_df=None):
                df = input_df
                df = self._add_ratio_features(df)
                df = self._add_growth_rates(df)
                df = self._add_lag_features(df)
                return df

        fe = MockFE(spark, sample_config)
        gold_df = fe.run(input_df=silver_df)
        for lag in sample_config["feature_engineering"]["lag_windows"]:
            assert f"totals_revenue_lag{lag}" in gold_df.columns


# ─────────────────────────────────────────────
# Anomaly detector tests
# ─────────────────────────────────────────────

class TestAnomalyDetector:
    def test_isolation_forest_output_shape(self, sample_config):
        from src.models.anomaly_detector import AnomalyDetector
        from sklearn.ensemble import IsolationForest
        import numpy as np

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 6))

        class MockDetector(AnomalyDetector):
            pass

        detector = MockDetector.__new__(MockDetector)
        detector.model_cfg = sample_config["models"]["anomaly_detector"]
        detector.contamination = 0.05
        detector.feature_cols = sample_config["models"]["anomaly_detector"]["features"]

        model, scores = detector._train_isolation_forest(X)
        assert len(scores) == 100
        assert isinstance(model, IsolationForest)

    def test_autoencoder_threshold_positive(self, sample_config):
        from src.models.anomaly_detector import AnomalyDetector
        import numpy as np

        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 6))

        detector = AnomalyDetector.__new__(AnomalyDetector)
        detector.model_cfg = sample_config["models"]["anomaly_detector"]
        detector.feature_cols = sample_config["models"]["anomaly_detector"]["features"]

        _, threshold, errors = detector._train_autoencoder(X)
        assert threshold > 0
        assert len(errors) == 80

    def test_ensemble_score_shape(self, sample_config):
        from src.models.anomaly_detector import AnomalyDetector
        import numpy as np

        detector = AnomalyDetector.__new__(AnomalyDetector)
        iso_scores = np.random.randn(50)
        ae_errors = np.abs(np.random.randn(50))
        result = detector._ensemble_score(iso_scores, ae_errors)
        assert len(result) == 50


# ─────────────────────────────────────────────
# Genetic Algorithm tests
# ─────────────────────────────────────────────

class TestResourceOptimizer:
    def test_random_individual_sums_to_one(self, spark, sample_config):
        from src.models.resource_optimizer import ResourceOptimizer
        optimizer = ResourceOptimizer(spark, sample_config)
        individual = optimizer._random_individual()
        assert abs(individual.sum() - 1.0) < 1e-6

    def test_repair_enforces_bounds(self, spark, sample_config):
        from src.models.resource_optimizer import ResourceOptimizer
        import numpy as np
        optimizer = ResourceOptimizer(spark, sample_config)
        raw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # violates all min shares
        repaired = optimizer._repair(raw.copy())
        bounds = sample_config["models"]["optimizer"]["constraint_bounds"]
        assert repaired[0] >= bounds["welfare_min_share"]
        assert repaired[1] >= bounds["highway_min_share"]
        assert repaired[2] >= bounds["police_min_share"]
        assert abs(repaired.sum() - 1.0) < 1e-5

    def test_evaluate_returns_float(self, spark, sample_config):
        from src.models.resource_optimizer import ResourceOptimizer
        optimizer = ResourceOptimizer(spark, sample_config)
        alloc = optimizer._random_individual()
        context = {"current_debt": 5e9, "total_revenue": 1e10, "fiscal_health_index": 0.6}
        score = optimizer._evaluate(alloc, 1e10, context)
        assert isinstance(score, float)

    def test_ga_runs_without_error(self, spark, sample_config):
        from src.models.resource_optimizer import ResourceOptimizer
        optimizer = ResourceOptimizer(spark, sample_config)
        context = {"current_debt": 5e9, "total_revenue": 1e10, "fiscal_health_index": 0.6}
        best, history = optimizer._run_ga(1e10, context)
        assert len(best) == 6
        assert abs(best.sum() - 1.0) < 1e-5
        assert len(history) == sample_config["models"]["optimizer"]["ga"]["generations"]

    def test_crossover_produces_two_children(self, spark, sample_config):
        from src.models.resource_optimizer import ResourceOptimizer
        import numpy as np
        optimizer = ResourceOptimizer(spark, sample_config)
        p1 = optimizer._random_individual()
        p2 = optimizer._random_individual()
        c1, c2 = optimizer._crossover(p1, p2)
        assert len(c1) == len(p1)
        assert len(c2) == len(p2)

    def test_mutation_stays_valid(self, spark, sample_config):
        from src.models.resource_optimizer import ResourceOptimizer
        optimizer = ResourceOptimizer(spark, sample_config)
        individual = optimizer._random_individual()
        mutated = optimizer._mutate(individual.copy(), rate=1.0)  # mutate all genes
        assert abs(mutated.sum() - 1.0) < 1e-5
        assert all(mutated >= 0)


# ─────────────────────────────────────────────
# MLflow utils tests (offline, no server needed)
# ─────────────────────────────────────────────

class TestMlflowUtils:
    def test_flatten_dict(self):
        from src.utils.mlflow_utils import _flatten
        nested = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        flat = _flatten(nested)
        assert flat == {"a.b": 1, "a.c.d": 2, "e": 3}

    def test_regression_metrics(self):
        from src.models.fiscal_forecaster import _regression_metrics
        import numpy as np
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([110.0, 190.0, 310.0, 390.0])
        metrics = _regression_metrics(y_true, y_pred)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert metrics["rmse"] > 0
        assert metrics["r2"] <= 1.0


# ─────────────────────────────────────────────
# Run directly in Databricks
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    print(result.stderr)
