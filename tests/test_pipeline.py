import numpy as np
import pandas as pd

from nyaya_gate.bootstrap import stratified_bootstrap_sample
from nyaya_gate.pipeline import run_cross_fitted_analysis


def _small_pipeline_config():
    return {
        "project": {"random_seed": 19},
        "data": {
            "modelling_variables": {
                "numeric": ["AGE"],
                "categorical": ["SEX"],
            },
            "aspirin": {"nominal_randomisation_probability": 0.5},
        },
        "cross_fitting": {"folds": 2, "primary_seed": 19},
        "estimators": {
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
                "random_state": 19,
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
                "random_state": 19,
            },
        },
        "gate": {
            "data_adequacy": {"essential_variables": ["AGE", "SEX"]},
            "reference_support": {"neighbours_per_treatment_arm": 2},
        },
    }


def test_pipeline_accepts_duplicate_source_ids_with_unique_draw_ids():
    rng = np.random.default_rng(21)
    n = 400
    age = rng.normal(70, 9, n)
    sex = rng.choice(["F", "M"], n)
    aspirin = rng.binomial(1, 0.5, n)
    linear = -0.5 + 0.02 * (age - 70) + 0.15 * (sex == "M") - 0.2 * aspirin
    outcome = rng.binomial(1, 1 / (1 + np.exp(-linear)))
    source = pd.DataFrame(
        {
            "study_row_id": np.arange(n),
            "aspirin": aspirin,
            "outcome": outcome,
            "AGE": age,
            "SEX": sex,
        }
    )
    sample = stratified_bootstrap_sample(source, random_state=22)
    assert sample["study_row_id"].duplicated().any()

    messages = []
    analysis = run_cross_fitted_analysis(
        sample,
        _small_pipeline_config(),
        fold_seed=23,
        estimator_seed=24,
        analysis_row_id=sample["bootstrap_row_id"].to_numpy(),
        model_n_jobs=1,
        progress=messages.append,
    )

    assert len(analysis.patient_results) == n
    assert analysis.patient_results["analysis_row_id"].is_unique
    assert np.isfinite(analysis.policy_risks["expected_risk"]).all()
    assert len(messages) == 2
