"""Prespecified treatment-effect estimators for the IST analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .preprocessing import FoldPreprocessor


def _binary_vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.int8)
    if result.ndim != 1 or not np.isin(result, [0, 1]).all():
        raise ValueError(f"{name} must be a binary vector.")
    return result


class GradientBoostedTLearner:
    """Separate gradient-boosted outcome models for both aspirin arms."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def fit(self, features: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> Self:
        x = np.asarray(features, dtype=float)
        a = _binary_vector(treatment, "treatment")
        y = _binary_vector(outcome, "outcome")
        if not (len(x) == len(a) == len(y)):
            raise ValueError("Features, treatment and outcome must have equal length.")
        self.models_: dict[int, HistGradientBoostingClassifier] = {}
        for arm in (0, 1):
            rows = a == arm
            if len(np.unique(y[rows])) != 2:
                raise ValueError(f"Treatment arm {arm} must contain both outcome classes.")
            model = HistGradientBoostingClassifier(
                learning_rate=float(self.config["learning_rate"]),
                max_iter=int(self.config["max_iter"]),
                max_leaf_nodes=int(self.config["max_leaf_nodes"]),
                min_samples_leaf=int(self.config["min_samples_leaf"]),
                l2_regularization=float(self.config["l2_regularization"]),
                random_state=int(self.config["random_state"]),
            )
            model.fit(x[rows], y[rows])
            self.models_[arm] = model
        return self

    def potential_outcome_risks(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=float)
        mu0 = self.models_[0].predict_proba(x)[:, 1]
        mu1 = self.models_[1].predict_proba(x)[:, 1]
        return mu0, mu1

    def effect(self, features: np.ndarray) -> np.ndarray:
        mu0, mu1 = self.potential_outcome_risks(features)
        return mu1 - mu0


class PenalisedLogisticInteractions:
    """L2 logistic model with every treatment-by-covariate interaction."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def fit(self, features: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> Self:
        x = np.asarray(features, dtype=float)
        a = _binary_vector(treatment, "treatment")
        y = _binary_vector(outcome, "outcome")
        if not (len(x) == len(a) == len(y)):
            raise ValueError("Features, treatment and outcome must have equal length.")
        self.scaler_ = StandardScaler().fit(x)
        scaled = self.scaler_.transform(x)
        if str(self.config["penalty"]).lower() != "l2":
            raise ValueError("The prespecified logistic penalty must be L2.")
        self.model_ = LogisticRegression(
            C=float(self.config["C"]),
            l1_ratio=0.0,
            solver=str(self.config["solver"]),
            max_iter=int(self.config["max_iter"]),
        )
        self.model_.fit(self._design(scaled, a), y)
        return self

    def potential_outcome_risks(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.scaler_.transform(np.asarray(features, dtype=float))
        zeros = np.zeros(len(x), dtype=np.int8)
        ones = np.ones(len(x), dtype=np.int8)
        mu0 = self.model_.predict_proba(self._design(x, zeros))[:, 1]
        mu1 = self.model_.predict_proba(self._design(x, ones))[:, 1]
        return mu0, mu1

    def effect(self, features: np.ndarray) -> np.ndarray:
        mu0, mu1 = self.potential_outcome_risks(features)
        return mu1 - mu0

    @staticmethod
    def _design(features: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        column = treatment.reshape(-1, 1).astype(float)
        return np.concatenate([column, features, features * column], axis=1)


class CausalForestRiskDifference:
    """CausalForestDML wrapper returning probability-scale risk differences."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def fit(self, features: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> Self:
        x = np.asarray(features, dtype=float)
        a = _binary_vector(treatment, "treatment")
        y = _binary_vector(outcome, "outcome")
        if not (len(x) == len(a) == len(y)):
            raise ValueError("Features, treatment and outcome must have equal length.")
        self.model_ = CausalForestDML(
            model_y=str(self.config.get("model_y", "auto")),
            model_t=str(self.config.get("model_t", "auto")),
            discrete_outcome=bool(self.config.get("discrete_outcome", True)),
            discrete_treatment=bool(self.config["discrete_treatment"]),
            cv=int(self.config.get("nuisance_cv", 2)),
            n_estimators=int(self.config["n_estimators"]),
            min_samples_leaf=int(self.config["min_samples_leaf"]),
            max_depth=self.config["max_depth"],
            max_samples=float(self.config["max_samples"]),
            honest=bool(self.config["honest"]),
            inference=bool(self.config["inference"]),
            subforest_size=int(self.config.get("subforest_size", 4)),
            n_jobs=int(self.config.get("n_jobs", -1)),
            random_state=int(self.config["random_state"]),
        )
        self.model_.fit(y, a, X=x)
        return self

    def effect(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.effect(np.asarray(features, dtype=float))).reshape(-1)

    def standard_error(self, features: np.ndarray) -> np.ndarray:
        if not bool(self.config["inference"]):
            raise ValueError("Causal-forest inference must be enabled for uncertainty.")
        inference = self.model_.effect_inference(np.asarray(features, dtype=float))
        return np.asarray(inference.stderr).reshape(-1)


@dataclass(frozen=True)
class EffectPredictions:
    causal_forest: np.ndarray
    causal_forest_standard_error: np.ndarray
    penalised_logistic: np.ndarray
    gradient_boosted_t_learner: np.ndarray
    outcome_risk_no_aspirin: np.ndarray
    outcome_risk_aspirin: np.ndarray

    def effect_matrix(self) -> np.ndarray:
        """Return effects in the order fixed by the stability set."""
        return np.column_stack(
            [
                self.causal_forest,
                self.penalised_logistic,
                self.gradient_boosted_t_learner,
            ]
        )


def fit_effect_estimators(
    training_data: pd.DataFrame,
    held_out_data: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    numeric_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    estimator_config: dict[str, Any],
) -> EffectPredictions:
    """Fit the three estimators on one training fold and predict held-out cases."""
    preprocessor = FoldPreprocessor(numeric_columns, categorical_columns)
    train_x = preprocessor.fit_transform(training_data)
    held_out_x = preprocessor.transform(held_out_data)

    forest = CausalForestRiskDifference(estimator_config["causal_forest"]).fit(
        train_x, treatment, outcome
    )
    logistic = PenalisedLogisticInteractions(
        estimator_config["penalised_logistic_interactions"]
    ).fit(train_x, treatment, outcome)
    t_learner = GradientBoostedTLearner(
        estimator_config["gradient_boosted_t_learner"]
    ).fit(train_x, treatment, outcome)
    mu0, mu1 = t_learner.potential_outcome_risks(held_out_x)

    return EffectPredictions(
        causal_forest=forest.effect(held_out_x),
        causal_forest_standard_error=forest.standard_error(held_out_x),
        penalised_logistic=logistic.effect(held_out_x),
        gradient_boosted_t_learner=mu1 - mu0,
        outcome_risk_no_aspirin=mu0,
        outcome_risk_aspirin=mu1,
    )
