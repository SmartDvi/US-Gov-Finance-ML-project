"""
gov_finance_ml/src/models/fiscal_forecaster.py

Model A — Predictive Fiscal Health (Time Series Forecasting)
============================================================
Insight: Use time-series models to forecast state revenue & expenditure
5+ years ahead, enabling early-warning budget signals.

Strategy:
- Prophet (seasonality-aware, interpretable, fast to tune)
- XGBoost with lag/rolling features (higher accuracy on structured data)
- Hyperparameter search tracked with MLflow
- Best model registered to UC Model Registry
"""

from __future__ import annotations

import logging
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.mlflow_utils import (
    log_json_artifact, log_metrics_dict, log_params_flat, managed_run,
    register_model,
)
from src.utils.spark_utils import read_delta

logger = logging.getLogger(__name__)


class FiscalForecaster:
    """
    Trains and evaluates fiscal revenue / expenditure forecasting models.

    Parameters
    ----------
    spark  : active SparkSession
    config : full project config dict
    """

    def __init__(self, spark: SparkSession, config: dict) -> None:
        self.spark = spark
        self.cfg = config
        self.model_cfg = config["models"]["fiscal_forecaster"]
        self.horizon = self.model_cfg["horizon_years"]

    # ── Public API ────────────────────────────────

    def run(self, df: Optional[DataFrame] = None) -> dict:
        """Train Prophet + XGBoost per state, log to MLflow, register best."""
        if df is None:
            df = read_delta(self.spark, self.cfg["data"]["gold_table"])

        df_train = df.filter(F.col("split") == "train")
        df_test = df.filter(F.col("split") == "test")

        # Convert to Pandas per state (manageable for ~50 states × 30 years)
        pdf_train = df_train.orderBy("state", "year").toPandas()
        pdf_test = df_test.orderBy("state", "year").toPandas()

        results = {}
        states = pdf_train["state"].unique()
        logger.info("Training fiscal forecasters for %d states", len(states))

        for target in ["totals_revenue", "totals_expenditure"]:
            results[target] = self._train_target(pdf_train, pdf_test, target)

        return results

    def _train_target(
        self, pdf_train: pd.DataFrame, pdf_test: pd.DataFrame, target: str
    ) -> dict:
        exp_path = f"{self.cfg['mlflow']['experiment_base']}/fiscal_forecasting"
        mlflow.set_experiment(exp_path)

        best_run_id = None
        best_rmse = float("inf")
        best_model_name = None

        # ── Prophet sweep ────────────────────────
        for cps in self.model_cfg["prophet"]["changepoint_prior_scale"]:
            for sps in self.model_cfg["prophet"]["seasonality_prior_scale"]:
                for mode in self.model_cfg["prophet"]["seasonality_mode"]:
                    run_name = f"prophet_{target}_cps{cps}_sps{sps}_{mode}"
                    with managed_run(run_name, tags={"model": "prophet", "target": target}) as run:
                        metrics, model = self._train_prophet(
                            pdf_train, pdf_test, target, cps, sps, mode
                        )
                        log_params_flat(
                            {"changepoint_prior_scale": cps,
                             "seasonality_prior_scale": sps,
                             "seasonality_mode": mode}
                        )
                        log_metrics_dict(metrics)
                        mlflow.sklearn.log_model(
                            model, "prophet_model",
                            signature=None,   # Prophet uses DataFrame I/O
                        )
                        if metrics["rmse"] < best_rmse:
                            best_rmse = metrics["rmse"]
                            best_run_id = run.info.run_id
                            best_model_name = "prophet_model"
                        logger.info("Prophet %s | RMSE=%.2f MAE=%.2f R2=%.3f",
                                    target, metrics["rmse"], metrics["mae"], metrics["r2"])

        # ── XGBoost sweep ────────────────────────
        for n_est in self.model_cfg["xgb"]["n_estimators"]:
            for depth in self.model_cfg["xgb"]["max_depth"]:
                for lr in self.model_cfg["xgb"]["learning_rate"]:
                    for ss in self.model_cfg["xgb"]["subsample"]:
                        run_name = f"xgb_{target}_n{n_est}_d{depth}_lr{lr}"
                        with managed_run(run_name, tags={"model": "xgboost", "target": target}) as run:
                            metrics, model, feat_imp = self._train_xgb(
                                pdf_train, pdf_test, target, n_est, depth, lr, ss
                            )
                            log_params_flat(
                                {"n_estimators": n_est, "max_depth": depth,
                                 "learning_rate": lr, "subsample": ss}
                            )
                            log_metrics_dict(metrics)
                            log_json_artifact(feat_imp, f"feature_importance_{target}.json")
                            mlflow.sklearn.log_model(model, "xgb_model")

                            if metrics["rmse"] < best_rmse:
                                best_rmse = metrics["rmse"]
                                best_run_id = run.info.run_id
                                best_model_name = "xgb_model"
                            logger.info("XGB %s | RMSE=%.2f MAE=%.2f R2=%.3f",
                                        target, metrics["rmse"], metrics["mae"], metrics["r2"])

        # ── Register champion ─────────────────────
        registered = f"{self.cfg['mlflow']['registered_model_prefix']}.fiscal_forecaster_{target}"
        if best_run_id:
            version = register_model(
                run_id=best_run_id,
                artifact_path=best_model_name,
                registered_name=registered,
                alias="champion",
                description=f"Best fiscal forecaster for {target} | best RMSE={best_rmse:.2f}",
            )
            logger.info("Champion model registered: %s v%s", registered, version)

        return {"best_rmse": best_rmse, "best_run_id": best_run_id}

    # ── Model trainers ────────────────────────────

    def _train_prophet(
        self,
        pdf_train: pd.DataFrame,
        pdf_test: pd.DataFrame,
        target: str,
        cps: float, sps: float, mode: str,
    ):
        from prophet import Prophet  # type: ignore

        # Aggregate to national level for simplicity (per-state is parallelisable with pandas_udf)
        train_agg = (
            pdf_train.groupby("year")[target].sum()
            .reset_index()
            .rename(columns={"year": "ds", target: "y"})
        )
        train_agg["ds"] = pd.to_datetime(train_agg["ds"], format="%Y")

        test_agg = (
            pdf_test.groupby("year")[target].sum()
            .reset_index()
            .rename(columns={"year": "ds", target: "y"})
        )
        test_agg["ds"] = pd.to_datetime(test_agg["ds"], format="%Y")

        model = Prophet(
            changepoint_prior_scale=cps,
            seasonality_prior_scale=sps,
            seasonality_mode=mode,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        model.fit(train_agg)

        future = model.make_future_dataframe(periods=len(test_agg), freq="Y")
        forecast = model.predict(future)
        y_pred = forecast.tail(len(test_agg))["yhat"].values
        y_true = test_agg["y"].values

        metrics = _regression_metrics(y_true, y_pred)
        return metrics, model

    def _train_xgb(
        self,
        pdf_train: pd.DataFrame,
        pdf_test: pd.DataFrame,
        target: str,
        n_est: int, depth: int, lr: float, subsample: float,
    ):
        from xgboost import XGBRegressor  # type: ignore

        feature_cols = _xgb_feature_cols(pdf_train, target)
        X_train = pdf_train[feature_cols].fillna(0)
        y_train = pdf_train[target].fillna(0)
        X_test = pdf_test[feature_cols].fillna(0)
        y_test = pdf_test[target].fillna(0)

        model = XGBRegressor(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            subsample=subsample,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        metrics = _regression_metrics(y_test.values, y_pred)

        feat_imp = dict(zip(feature_cols, model.feature_importances_.tolist()))
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x: -x[1])[:20])

        return metrics, model, feat_imp

    # ── Pandas UDF for scalable per-state training ─

    def train_per_state_xgb_udf(self, df: DataFrame, target: str) -> DataFrame:
        """
        Distributed per-state training using PySpark Pandas UDF.
        Returns a DataFrame of (state, rmse, mae, r2) metrics.
        """
        from pyspark.sql.types import (
            DoubleType, StringType, StructField, StructType,
        )

        result_schema = StructType([
            StructField("state", StringType()),
            StructField("rmse", DoubleType()),
            StructField("mae", DoubleType()),
            StructField("r2", DoubleType()),
        ])

        model_cfg = self.model_cfg
        cutoff = self.cfg["data"]["train_cutoff_year"]

        @F.pandas_udf(result_schema, F.PandasUDFType.GROUPED_MAP)
        def train_state(pdf: pd.DataFrame) -> pd.DataFrame:
            from xgboost import XGBRegressor
            train = pdf[pdf["year"] <= cutoff]
            test = pdf[pdf["year"] > cutoff]
            if len(test) < 2 or len(train) < 5:
                return pd.DataFrame(
                    {"state": [pdf["state"].iloc[0]], "rmse": [0.0], "mae": [0.0], "r2": [0.0]}
                )
            feat_cols = _xgb_feature_cols(train, target)
            model = XGBRegressor(
                n_estimators=model_cfg["xgb"]["n_estimators"][1],
                max_depth=model_cfg["xgb"]["max_depth"][1],
                random_state=42,
            )
            model.fit(train[feat_cols].fillna(0), train[target].fillna(0))
            preds = model.predict(test[feat_cols].fillna(0))
            m = _regression_metrics(test[target].fillna(0).values, preds)
            return pd.DataFrame(
                {"state": [pdf["state"].iloc[0]], **{k: [v] for k, v in m.items()}}
            )

        return df.groupBy("state").apply(train_state)


# ── Helpers ───────────────────────────────────────

def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def _xgb_feature_cols(pdf: pd.DataFrame, target: str) -> list[str]:
    exclude = {target, "state", "split", "_ingested_at", "_row_id", "data_quality_score"}
    str_cols = {c for c in pdf.columns if pdf[c].dtype == object}
    return [
        c for c in pdf.columns
        if c not in exclude and c not in str_cols and pdf[c].dtype in (float, int, "float64", "int64")
    ]
