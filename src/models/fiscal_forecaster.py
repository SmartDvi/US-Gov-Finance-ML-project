"""
Model A — Next-year revenue forecaster (Spark ML RandomForestRegressor).

Business question: "How much general revenue (taxes, federal aid, charges)
will each state collect next year?" This is the basis for budget planning
and early warning of shortfalls. Total revenue is not used as the target
because it includes pension-fund investment returns, which follow the
stock market rather than the state's economy.

How it works
------------
* Label = next year's revenue GROWTH RATE, not the dollar amount. Tree models
  cannot predict values larger than anything seen in training, and revenue
  keeps growing over time, so predicting dollars would fail on recent years.
  Growth rates are also comparable across big and small states.
  Forecast in dollars = this year's revenue * (1 + predicted growth).

* Features only use information known in the current year (no leakage).

* Time-based split (never random for time series):
      train      : target year <  validation_start_year
      validation : used to choose hyper-parameters
      test       : target year >= test_start_year  (touched once, at the end)

* Random forest: an average of many shallow decision trees. With only ~500
  training rows it overfits less than gradient boosting and has few
  hyper-parameters to tune.

* The model is always reported next to two simple baselines, so we can see
  whether it adds value:
      naive          : revenue stays the same next year (growth = 0)
      average growth : every state grows at the historical average rate
"""

from __future__ import annotations

import itertools
import logging

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.features.feature_engineering import state_window
from src.utils.spark_utils import safe_divide

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    # momentum
    "totals_revenue_growth",
    "totals_revenue_growth_prev",
    "totals_general_revenue_growth",
    "totals_tax_growth",
    "totals_expenditure_growth",
    "totals_debt_at_end_of_fiscal_year_growth",
    "national_revenue_growth",
    # structure of the state's finances
    "debt_to_revenue_ratio",
    "expenditure_to_revenue_ratio",
    "interest_burden",
    "intergovernmental_dependency",
    "tax_share_of_general_revenue",
    "insurance_trust_share_of_revenue",
]


def add_label(df: DataFrame, target: str) -> DataFrame:
    """Add next year's value and growth rate (null when next year is missing)."""
    next_year = F.lead("year").over(state_window())
    next_value = F.when(next_year == F.col("year") + 1, F.lead(target).over(state_window()))
    return (
        df.withColumn("target_year", F.col("year") + 1)
        .withColumn("current_value", F.col(target))
        .withColumn("next_value", next_value)
        .withColumn("label", safe_divide(F.col("next_value") - F.col(target), F.abs(F.col(target))))
    )


class FiscalForecaster:
    def __init__(self, config: dict) -> None:
        cfg = config["forecaster"]
        self.target = cfg["target"]
        self.validation_start = cfg["validation_start_year"]
        self.test_start = cfg["test_start_year"]
        self.seed = cfg["seed"]
        self.param_grid = [
            {"max_depth": d, "num_trees": n}
            for d, n in itertools.product(cfg["param_grid"]["max_depth"], cfg["param_grid"]["num_trees"])
        ]
        self.evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")

    # ── Public API ──────────────────────────────────────────────

    def run(self, features_df: DataFrame) -> dict:
        """Tune, evaluate on the test years, refit on all data, forecast next year."""
        df = add_label(features_df, self.target)
        labelled = df.dropna(subset=FEATURE_COLS + ["label"]).cache()

        train = labelled.filter(F.col("target_year") < self.validation_start)
        validation = labelled.filter(
            (F.col("target_year") >= self.validation_start) & (F.col("target_year") < self.test_start)
        )
        train_and_validation = labelled.filter(F.col("target_year") < self.test_start)
        test = labelled.filter(F.col("target_year") >= self.test_start)
        logger.info(
            "Rows -> train=%d validation=%d test=%d",
            train.count(), validation.count(), test.count(),
        )

        # 1) Choose hyper-parameters on the validation years
        best_params, tuning_results = self._tune(train, validation)

        # 2) Refit on train + validation, evaluate ONCE on the unseen test years
        model = self._build_pipeline(**best_params).fit(train_and_validation)
        test_predictions = self._with_revenue_forecast(model.transform(test))
        average_growth = train_and_validation.agg(F.avg("label")).first()[0]
        test_metrics = {
            "random_forest": self._metrics(test_predictions),
            "baseline_naive_no_change": self._metrics(
                self._with_revenue_forecast(test.withColumn("prediction", F.lit(0.0)))
            ),
            "baseline_average_growth": self._metrics(
                self._with_revenue_forecast(test.withColumn("prediction", F.lit(average_growth)))
            ),
        }

        # 3) Production model: refit on ALL labelled years, forecast the year after the data ends
        final_model = self._build_pipeline(**best_params).fit(labelled)
        latest_year = df.agg(F.max("year")).first()[0]
        latest = df.filter(F.col("year") == latest_year).dropna(subset=FEATURE_COLS)
        forecast = self._with_revenue_forecast(final_model.transform(latest)).select(
            "state",
            F.col("year").alias("base_year"),
            F.col("target_year").alias("forecast_year"),
            F.col("current_value").alias(f"{self.target}_base_year"),
            F.col("prediction").alias("predicted_growth"),
            F.col("forecast_value").alias(f"{self.target}_forecast"),
        )

        labelled.unpersist()
        return {
            "model": final_model,
            "forecast": forecast,
            "test_predictions": test_predictions.select(
                "state", "year", "target_year", "current_value", "next_value",
                F.col("label").alias("actual_growth"),
                F.col("prediction").alias("predicted_growth"),
                "forecast_value",
            ),
            "metrics": {
                "target": self.target,
                "best_params": best_params,
                "tuning_results": tuning_results,
                "test": test_metrics,
                "feature_importance": self._feature_importance(final_model),
            },
        }

    # ── Internals ───────────────────────────────────────────────

    def _build_pipeline(self, max_depth: int, num_trees: int) -> Pipeline:
        assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
        forest = RandomForestRegressor(
            featuresCol="features",
            labelCol="label",
            maxDepth=max_depth,
            numTrees=num_trees,
            seed=self.seed,
        )
        return Pipeline(stages=[assembler, forest])

    def _tune(self, train: DataFrame, validation: DataFrame) -> tuple[dict, list[dict]]:
        results = []
        for params in self.param_grid:
            model = self._build_pipeline(**params).fit(train)
            mae = self.evaluator.evaluate(model.transform(validation), {self.evaluator.metricName: "mae"})
            results.append({**params, "validation_mae_pct_points": round(mae * 100, 2)})
            logger.info("Tuning %s -> validation MAE %.2f pp", params, mae * 100)
        best = min(results, key=lambda r: r["validation_mae_pct_points"])
        best_params = {"max_depth": best["max_depth"], "num_trees": best["num_trees"]}
        logger.info("Best hyper-parameters: %s", best_params)
        return best_params, results

    @staticmethod
    def _with_revenue_forecast(predictions: DataFrame) -> DataFrame:
        """Convert a predicted growth rate back to dollars."""
        return predictions.withColumn(
            "forecast_value", F.col("current_value") * (1 + F.col("prediction"))
        )

    def _metrics(self, predictions: DataFrame) -> dict:
        """Growth-rate errors (MAE/RMSE, in percentage points) and dollar MAPE."""
        mae = self.evaluator.evaluate(predictions, {self.evaluator.metricName: "mae"})
        rmse = self.evaluator.evaluate(predictions, {self.evaluator.metricName: "rmse"})
        mape = predictions.agg(
            F.avg(F.abs(F.col("forecast_value") - F.col("next_value")) / F.abs(F.col("next_value")))
        ).first()[0]
        return {
            "growth_mae_pct_points": round(mae * 100, 2),
            "growth_rmse_pct_points": round(rmse * 100, 2),
            "revenue_mape_pct": round(mape * 100, 2),
        }

    @staticmethod
    def _feature_importance(model: PipelineModel) -> dict:
        importances = model.stages[-1].featureImportances.toArray()
        ranked = sorted(zip(FEATURE_COLS, importances), key=lambda kv: -kv[1])
        return {name: round(float(value), 4) for name, value in ranked}
