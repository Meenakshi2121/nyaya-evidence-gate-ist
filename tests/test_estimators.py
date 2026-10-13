import numpy as np
import pandas as pd

from nyaya_gate.estimators import (
    CausalForestRiskDifference,
    GradientBoostedTLearner,
    PenalisedLogisticInteractions,
    fit_effect_estimators,
)
from nyaya_gate.preprocessing import FoldPreprocessor


def _synthetic_data(seed: int = 4, n: int = 320):
    rng = np.random.default_rng(seed)
    age = rng.normal(70, 9, n)
    sex = rng.choice(["F", "M"], n)
    treatment = rng.binomial(1, 0.5, n)
    linear = -0.8 + 0.025 * (age - 70) + 0.2 * (sex == "M")
    linear += treatment * (-0.35 + 0.012 * (age - 70))
    risk = 1 / (1 + np.exp(-linear))
    outcome = rng.binomial(1, risk)
    data = pd.DataFrame({"age": age, "sex": sex})
    features = FoldPreprocessor(("age",), ("sex",)).fit_transform(data)
    return features, treatment, outcome


def test_probability_scale_effects_from_logistic_and_t_learner():
    features, treatment, outcome = _synthetic_data()
    logistic = PenalisedLogisticInteractions(
        {"penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 1000}
    ).fit(features, treatment, outcome)
    t_learner = GradientBoostedTLearner(
        {
            "learning_rate": 0.05,
            "max_iter": 40,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 10,
            "l2_regularization": 1.0,
            "random_state": 5,
        }
    ).fit(features, treatment, outcome)
    for effects in (logistic.effect(features[:12]), t_learner.effect(features[:12])):
        assert effects.shape == (12,)
        assert np.isfinite(effects).all()
        assert np.all((effects >= -1) & (effects <= 1))


def test_causal_forest_returns_effects_and_standard_errors():
    features, treatment, outcome = _synthetic_data(n=400)
    forest = CausalForestRiskDifference(
        {
            "model_y": "auto",
            "model_t": "auto",
            "discrete_outcome": True,
            "nuisance_cv": 2,
            "n_estimators": 40,
            "min_samples_leaf": 5,
            "max_depth": None,
            "max_samples": 0.45,
            "honest": True,
            "inference": True,
            "discrete_treatment": True,
            "subforest_size": 4,
            "n_jobs": 1,
            "random_state": 6,
        }
    ).fit(features, treatment, outcome)
    effects = forest.effect(features[:10])
    standard_errors = forest.standard_error(features[:10])
    assert effects.shape == standard_errors.shape == (10,)
    assert np.isfinite(effects).all()
    assert np.isfinite(standard_errors).all()
    assert np.all(standard_errors >= 0)


def test_combined_estimator_runner_returns_aligned_held_out_predictions():
    features, treatment, outcome = _synthetic_data(n=420)
    columns = tuple(f"x{position}" for position in range(features.shape[1]))
    data = pd.DataFrame(features, columns=columns)
    config = {
        "causal_forest": {
            "model_y": "auto",
            "model_t": "auto",
            "discrete_outcome": True,
            "nuisance_cv": 2,
            "n_estimators": 40,
            "min_samples_leaf": 5,
            "max_depth": None,
            "max_samples": 0.45,
            "honest": True,
            "inference": True,
            "discrete_treatment": True,
            "subforest_size": 4,
            "n_jobs": 1,
            "random_state": 6,
        },
        "penalised_logistic_interactions": {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
        },
        "gradient_boosted_t_learner": {
            "learning_rate": 0.05,
            "max_iter": 40,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 10,
            "l2_regularization": 1.0,
            "random_state": 5,
        },
    }
    split = 360
    predictions = fit_effect_estimators(
        data.iloc[:split],
        data.iloc[split:],
        treatment[:split],
        outcome[:split],
        numeric_columns=columns,
        categorical_columns=(),
        estimator_config=config,
    )
    assert predictions.effect_matrix().shape == (len(data) - split, 3)
    assert predictions.causal_forest_standard_error.shape == (len(data) - split,)
    assert np.isfinite(predictions.effect_matrix()).all()
