"""
Unit tests for each pipeline stage. Run with:  uv run pytest
"""

from __future__ import annotations

import pytest
from pyspark.sql import functions as F

from src.data.ingestion import clean, sanitize_column_name, validate
from src.features.feature_engineering import build_features
from src.models.anomaly_detector import AnomalyDetector, audit_report
from src.models.fiscal_forecaster import FiscalForecaster, add_label
from src.models.state_clustering import StateClusteringModel, get_peer_states
from tests.conftest import STATES


def _row(df, state, year):
    return df.filter((F.col("state") == state) & (F.col("year") == year)).first()


# ── Ingestion ─────────────────────────────────────────────────────

class TestIngestion:
    @pytest.mark.parametrize("raw, expected", [
        ("Totals.Capital outlay", "totals_capital_outlay"),
        ("Totals. Debt at end of fiscal year", "totals_debt_at_end_of_fiscal_year"),
        ("Totals.Insurance trust  revenue", "totals_insurance_trust_revenue"),
        ("Details.Welfare.Welfare Institution Total Expenditure",
         "details_welfare_welfare_institution_total_expenditure"),
    ])
    def test_sanitize_column_name(self, raw, expected):
        assert sanitize_column_name(raw) == expected

    def test_clean_casts_types_and_drops_national_total(self, spark):
        raw = spark.createDataFrame(
            [(" alabama ", "1992", "100"), ("UNITED STATES", "1992", "9999")],
            ["State", "Year", "Totals.Revenue"],
        )
        result = clean(raw, exclude_states=["United States"])
        assert result.columns == ["state", "year", "totals_revenue"]
        assert dict(result.dtypes) == {"state": "string", "year": "int", "totals_revenue": "double"}
        assert [r["state"] for r in result.collect()] == ["ALABAMA"]

    def test_validate_passes_on_good_data(self, clean_df):
        validate(clean_df)

    def test_validate_rejects_duplicates(self, clean_df):
        with pytest.raises(ValueError, match="duplicated"):
            validate(clean_df.union(clean_df.limit(1)))

    def test_validate_rejects_missing_columns(self, clean_df):
        with pytest.raises(ValueError, match="Missing required columns"):
            validate(clean_df.drop("totals_revenue"))


# ── Feature engineering ──────────────────────────────────────────

class TestFeatures:
    def test_growth_is_correct_for_consecutive_years(self, clean_df):
        features = build_features(clean_df)
        prev, curr = _row(clean_df, "ALPHA", 1999), _row(features, "ALPHA", 2000)
        expected = curr["totals_revenue"] / prev["totals_revenue"] - 1
        assert curr["totals_revenue_growth"] == pytest.approx(expected)

    def test_no_growth_across_the_2005_2011_gap(self, clean_df):
        """2012 follows 2004 in the data; that is NOT a one-year change."""
        features = build_features(clean_df)
        row_2012 = _row(features, "ALPHA", 2012)
        assert row_2012["totals_revenue_growth"] is None
        assert _row(features, "ALPHA", 2013)["totals_revenue_growth_prev"] is None

    def test_shares_are_between_0_and_1(self, clean_df):
        features = build_features(clean_df)
        bad = features.filter((F.col("welfare_share") < 0) | (F.col("welfare_share") > 1)).count()
        assert bad == 0

    def test_division_by_zero_gives_null(self, clean_df):
        zero_rev = clean_df.withColumn("totals_general_revenue", F.lit(0.0))
        features = build_features(zero_rev)
        assert features.filter(F.col("debt_to_revenue_ratio").isNotNull()).count() == 0


# ── Forecaster ───────────────────────────────────────────────────

class TestForecaster:
    def test_label_is_next_year_growth_with_no_leakage(self, clean_df):
        labelled = add_label(build_features(clean_df), "totals_revenue")
        this_year, next_year = _row(clean_df, "BRAVO", 2000), _row(clean_df, "BRAVO", 2001)
        row = _row(labelled, "BRAVO", 2000)
        assert row["label"] == pytest.approx(next_year["totals_revenue"] / this_year["totals_revenue"] - 1)
        # No label when the next year is missing (gap) or when the data ends
        assert _row(labelled, "BRAVO", 2004)["label"] is None
        assert _row(labelled, "BRAVO", 2019)["label"] is None

    def test_run_produces_metrics_and_one_forecast_per_state(self, clean_df, config):
        result = FiscalForecaster(config).run(build_features(clean_df))
        forecast = result["forecast"]
        assert forecast.count() == len(STATES)
        assert forecast.select(F.min("forecast_year")).first()[0] == 2020
        assert set(result["metrics"]["test"]) == {
            "random_forest", "baseline_naive_no_change", "baseline_average_growth"
        }


# ── Clustering ───────────────────────────────────────────────────

class TestClustering:
    def test_one_cluster_per_state_and_peers_exclude_self(self, clean_df, config):
        result = StateClusteringModel(config).run(build_features(clean_df))
        clusters = result["state_clusters"]
        assert clusters.count() == len(STATES)
        assert result["metrics"]["selected_k"] in config["clustering"]["k_range"]
        peers = get_peer_states(clusters, "alpha")
        assert "ALPHA" not in peers

    def test_unknown_state_raises(self, clean_df, config):
        clusters = StateClusteringModel(config).run(build_features(clean_df))["state_clusters"]
        with pytest.raises(ValueError):
            get_peer_states(clusters, "ATLANTIS")


# ── Anomaly detection ────────────────────────────────────────────

class TestAnomalyDetector:
    def test_injected_spike_is_flagged_with_reason(self, clean_df, config):
        # Capital outlay jumps 10x in one year for one state
        spiked = clean_df.withColumn(
            "totals_capital_outlay",
            F.when((F.col("state") == "CHARLIE") & (F.col("year") == 2015),
                   F.col("totals_capital_outlay") * 10).otherwise(F.col("totals_capital_outlay")),
        )
        scores = AnomalyDetector(config).run(build_features(spiked))["scores"]
        flagged = _row(audit_report(scores), "CHARLIE", 2015)
        assert flagged is not None
        assert "totals_capital_outlay_growth" in flagged["reasons"]

    def test_score_before_fit_raises(self, clean_df, config):
        with pytest.raises(RuntimeError):
            AnomalyDetector(config).score(clean_df)
