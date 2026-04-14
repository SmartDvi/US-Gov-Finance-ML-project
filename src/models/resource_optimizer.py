"""
gov_finance_ml/src/models/resource_optimizer.py

Model D — Multi-Objective Resource Optimization
================================================
Insight: Find the Pareto-optimal allocation of general revenue across
Welfare, Highways, Police, Parks, and Debt Service — simultaneously
minimising debt growth and maximising welfare + infrastructure outcomes.

Strategy:
- Genetic Algorithm (DEAP) for Pareto frontier exploration
- Reward function: weighted composite of fiscal objectives
- Constraint enforcement: spend-share guardrails per config
- Results: optimal allocation table per state + year
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.utils.mlflow_utils import (
    log_json_artifact, log_metrics_dict, log_params_flat, managed_run,
    register_model,
)
from src.utils.spark_utils import read_delta, write_delta

logger = logging.getLogger(__name__)


# ── Budget categories (order matters for the chromosome) ─────────────
BUDGET_CATEGORIES = [
    "welfare",
    "highways",
    "police",
    "parks",
    "debt_service",
    "other",          # residual — absorbs remainder
]


class ResourceOptimizer:
    """
    Genetic Algorithm-based budget optimizer.

    Chromosome: a 6-element array of allocation fractions summing to 1.
    Fitness: multi-objective → maximise weighted composite score subject to constraints.
    """

    RESULT_TABLE_SUFFIX = "_optimal_allocations"

    def __init__(self, spark: SparkSession, config: dict) -> None:
        self.spark = spark
        self.cfg = config
        self.model_cfg = config["models"]["optimizer"]
        self.weights = self.model_cfg["objective_weights"]
        self.bounds = self.model_cfg["constraint_bounds"]
        self.ga_cfg = self.model_cfg["ga"]

    # ── Public API ─────────────────────────────────

    def run(self, df: Optional[DataFrame] = None) -> DataFrame:
        """Run GA optimisation for the latest year of data per state."""
        if df is None:
            df = read_delta(self.spark, self.cfg["data"]["gold_table"])

        # Get latest year stats per state for the optimisation context
        max_year = df.agg(F.max("year")).collect()[0][0]
        pdf = (
            df.filter(F.col("year") == max_year)
            .select(
                "state", "year",
                "totals_general_revenue",
                "totals_debt_at_end_of_fiscal_year",
                "totals_revenue",
                "welfare_expenditure_share",
                "highway_expenditure_share",
                "fiscal_health_index",
            )
            .toPandas()
        )

        exp_path = f"{self.cfg['mlflow']['experiment_base']}/resource_optimization"
        mlflow.set_experiment(exp_path)

        results = []
        for _, row in pdf.iterrows():
            state = row["state"]
            budget = row.get("totals_general_revenue", 0) or 0
            if budget <= 0:
                continue

            with managed_run(
                f"ga_optimizer_{state}",
                tags={"model": "genetic_algorithm", "state": state},
                nested=False,
            ) as run:
                log_params_flat({
                    "population_size": self.ga_cfg["population_size"],
                    "generations": self.ga_cfg["generations"],
                    "crossover_rate": self.ga_cfg["crossover_rate"],
                    "mutation_rate": self.ga_cfg["mutation_rate"],
                    "state": state,
                    "year": max_year,
                })

                context = {
                    "current_debt": row.get("totals_debt_at_end_of_fiscal_year", 0) or 0,
                    "total_revenue": row.get("totals_revenue", budget) or budget,
                    "fiscal_health_index": row.get("fiscal_health_index", 0.5) or 0.5,
                }

                best_alloc, fitness_history = self._run_ga(budget, context)
                score = self._evaluate(best_alloc, budget, context)

                log_metrics_dict({
                    "best_fitness": float(score),
                    "welfare_share": float(best_alloc[0]),
                    "highway_share": float(best_alloc[1]),
                    "police_share": float(best_alloc[2]),
                    "parks_share": float(best_alloc[3]),
                    "debt_service_share": float(best_alloc[4]),
                    "other_share": float(best_alloc[5]),
                })
                log_json_artifact(
                    {"fitness_history": fitness_history[-20:]},
                    f"fitness_history_{state}.json",
                )

                alloc_dict = dict(zip(BUDGET_CATEGORIES, best_alloc))
                results.append({
                    "state": state,
                    "year": max_year,
                    "total_budget": budget,
                    "fitness_score": score,
                    **{f"optimal_{cat}_share": share for cat, share in alloc_dict.items()},
                    **{f"optimal_{cat}_amount": share * budget for cat, share in alloc_dict.items()},
                    "run_id": run.info.run_id,
                })

        pdf_results = pd.DataFrame(results)
        result_spark = self.spark.createDataFrame(pdf_results)
        result_table = self.cfg["data"]["gold_table"] + self.RESULT_TABLE_SUFFIX
        write_delta(result_spark, result_table, mode="overwrite", partition_by=["state"])
        logger.info("Optimal allocations written → %s", result_table)
        return result_spark

    def compare_actual_vs_optimal(self, df: Optional[DataFrame] = None) -> pd.DataFrame:
        """Compare current spend shares against GA-recommended allocations."""
        gold_df = df or read_delta(self.spark, self.cfg["data"]["gold_table"])
        result_table = self.cfg["data"]["gold_table"] + self.RESULT_TABLE_SUFFIX
        opt_df = read_delta(self.spark, result_table)

        max_year = gold_df.agg(F.max("year")).collect()[0][0]
        actual = gold_df.filter(F.col("year") == max_year).select(
            "state", "welfare_expenditure_share", "highway_expenditure_share",
            "fiscal_health_index",
        )
        joined = opt_df.join(actual, on="state", how="left").toPandas()
        joined["welfare_gap"] = joined["optimal_welfare_share"] - joined["welfare_expenditure_share"]
        joined["highway_gap"] = joined["optimal_highway_share"] - joined["highway_expenditure_share"]
        return joined[["state", "welfare_gap", "highway_gap", "fitness_score", "fiscal_health_index"]]

    # ── Genetic Algorithm ─────────────────────────

    def _run_ga(self, budget: float, context: dict):
        pop_size = self.ga_cfg["population_size"]
        n_gen = self.ga_cfg["generations"]
        cr = self.ga_cfg["crossover_rate"]
        mr = self.ga_cfg["mutation_rate"]
        n_genes = len(BUDGET_CATEGORIES)

        # Initialise population
        population = [self._random_individual() for _ in range(pop_size)]
        fitness_history = []

        for gen in range(n_gen):
            # Evaluate
            fitnesses = [self._evaluate(ind, budget, context) for ind in population]
            best_fit = max(fitnesses)
            fitness_history.append(best_fit)

            # Selection (tournament)
            selected = [self._tournament_select(population, fitnesses) for _ in range(pop_size)]

            # Crossover + mutation
            offspring = []
            for i in range(0, pop_size, 2):
                p1 = selected[i]
                p2 = selected[min(i + 1, pop_size - 1)]
                if random.random() < cr:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                offspring.append(self._mutate(c1, mr))
                offspring.append(self._mutate(c2, mr))

            # Elitism: keep best individual
            best_idx = int(np.argmax(fitnesses))
            offspring[0] = population[best_idx].copy()
            population = offspring[:pop_size]

            if gen % 25 == 0:
                logger.debug("GA gen=%d | best_fitness=%.4f", gen, best_fit)

        final_fitnesses = [self._evaluate(ind, budget, context) for ind in population]
        best_idx = int(np.argmax(final_fitnesses))
        return population[best_idx], fitness_history

    def _random_individual(self) -> np.ndarray:
        """Random allocation vector summing to 1 within constraints."""
        n = len(BUDGET_CATEGORIES)
        raw = np.random.dirichlet(np.ones(n))
        return self._repair(raw)

    def _repair(self, alloc: np.ndarray) -> np.ndarray:
        """Enforce min/max share constraints via clipping + renormalisation."""
        b = self.bounds
        alloc[0] = np.clip(alloc[0], b["welfare_min_share"], b["welfare_max_share"])
        alloc[1] = np.clip(alloc[1], b["highway_min_share"], b["highway_max_share"])
        alloc[2] = np.clip(alloc[2], b["police_min_share"], b["police_max_share"])
        alloc[3] = np.clip(alloc[3], 0.01, 0.10)   # parks
        alloc[4] = np.clip(alloc[4], 0.03, 0.20)   # debt service
        alloc[5] = max(0.0, 1.0 - alloc[:5].sum())  # residual
        total = alloc.sum()
        return alloc / (total + 1e-8)

    def _evaluate(self, alloc: np.ndarray, budget: float, context: dict) -> float:
        """
        Multi-objective fitness function.
        Maximise: social welfare + infrastructure quality
        Minimise: debt growth + interest burden
        """
        welfare_share, highway_share, police_share, parks_share, debt_share, other = alloc
        w = self.weights
        fhi = context.get("fiscal_health_index", 0.5)

        # Social objective: welfare + parks
        social_score = (welfare_share * 3.0 + parks_share * 1.5) * w["maximize_welfare"]

        # Infrastructure objective: highways
        infra_score = highway_share * 2.0 * w["maximize_infrastructure"]

        # Safety: police with diminishing returns
        safety_score = (min(police_share, 0.10) * 2.0) * w["maximize_public_safety"]

        # Debt penalty: penalise high debt service
        debt_penalty = debt_share * 1.5 * w["minimize_debt"]

        # Fiscal health bonus
        fhi_bonus = fhi * 0.2

        fitness = social_score + infra_score + safety_score - debt_penalty + fhi_bonus

        # Hard constraint violation penalties
        if welfare_share < self.bounds["welfare_min_share"]:
            fitness -= 0.5
        if highway_share < self.bounds["highway_min_share"]:
            fitness -= 0.3

        return float(fitness)

    def _tournament_select(self, population, fitnesses, k: int = 3) -> np.ndarray:
        candidates = random.sample(range(len(population)), min(k, len(population)))
        best = max(candidates, key=lambda i: fitnesses[i])
        return population[best].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray):
        """Single-point crossover on Dirichlet-normalised arrays."""
        pt = random.randint(1, len(p1) - 1)
        c1 = np.concatenate([p1[:pt], p2[pt:]])
        c2 = np.concatenate([p2[:pt], p1[pt:]])
        return self._repair(c1), self._repair(c2)

    def _mutate(self, individual: np.ndarray, rate: float) -> np.ndarray:
        """Gaussian perturbation on each gene with probability=rate."""
        for i in range(len(individual)):
            if random.random() < rate:
                individual[i] += np.random.normal(0, 0.02)
        return self._repair(np.abs(individual))
