"""Reusable cross-fitted analysis pipeline for point and bootstrap runs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import numpy as np
import pandas as pd

from .adequacy import data_adequacy_check
from .analysis import PrimaryAnalysis, assemble_primary_analysis
from .crossfit import (
    make_grouped_stratified_folds,
    make_stratified_folds,
)
from .estimators import fit_effect_estimators
from .support import GowerReferenceSupport

ProgressCallback = Callable[[str], None]


def run_cross_fitted_analysis(
    data: pd.DataFrame,
    config: dict[str, Any],
    *,
    fold_seed: int | None = None,
    estimator_seed: int | None = None,
    analysis_row_id: np.ndarray | None = None,
    fold_group_id: np.ndarray | None = None,
    model_n_jobs: int | None = None,
    progress: ProgressCallback | None = None,
) -> PrimaryAnalysis:
    """Fit every model on training folds and analyse held-out predictions.

    ``analysis_row_id`` must be unique within the supplied data. It is used for
    deterministic matched-coverage tie-breaking.

    ``fold_group_id`` may identify repeated draws from the same source
    participant. All rows sharing an identifier stay in one cross-fitting fold.
    """
    modelling = config["data"]["modelling_variables"]
    numeric = tuple(modelling["numeric"])
    categorical = tuple(modelling["categorical"])
    required = {
        "study_row_id",
        "aspirin",
        "outcome",
        *numeric,
        *categorical,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Prepared analysis columns are missing: {missing}")

    treatment = data["aspirin"].to_numpy(dtype=np.int8)
    outcome = data["outcome"].to_numpy(dtype=np.int8)
    if analysis_row_id is None:
        row_id = data["study_row_id"].to_numpy()
    else:
        row_id = np.asarray(analysis_row_id)
    if row_id.ndim != 1 or len(row_id) != len(data):
        raise ValueError("analysis_row_id must contain one value per row.")
    if len(np.unique(row_id)) != len(row_id):
        raise ValueError("analysis_row_id values must be unique.")

    crossfit = config["cross_fitting"]
    selected_fold_seed = (
        int(crossfit["primary_seed"]) if fold_seed is None else int(fold_seed)
    )
    if fold_group_id is None:
        folds = make_stratified_folds(
            treatment,
            outcome,
            n_splits=int(crossfit["folds"]),
            random_state=selected_fold_seed,
            participant_id=row_id,
        )
    else:
        folds = make_grouped_stratified_folds(
            treatment,
            outcome,
            n_splits=int(crossfit["folds"]),
            random_state=selected_fold_seed,
            group_id=np.asarray(fold_group_id),
        )

    estimator_config = deepcopy(config["estimators"])
    if estimator_seed is not None:
        estimator_config["causal_forest"]["random_state"] = int(estimator_seed)
        estimator_config["gradient_boosted_t_learner"]["random_state"] = int(
            estimator_seed
        )
    if model_n_jobs is not None:
        estimator_config["causal_forest"]["n_jobs"] = int(model_n_jobs)

    n = len(data)
    reference_supported = np.empty(n, dtype=np.int8)
    reference_score = np.empty(n, dtype=float)
    causal_forest_effect = np.empty(n, dtype=float)
    causal_forest_standard_error = np.empty(n, dtype=float)
    penalised_logistic_effect = np.empty(n, dtype=float)
    t_learner_effect = np.empty(n, dtype=float)
    outcome_risk_no_aspirin = np.empty(n, dtype=float)
    outcome_risk_aspirin = np.empty(n, dtype=float)
    support_thresholds: dict[int, float] = {}

    support_config = config["gate"]["reference_support"]
    support_n_jobs = -1 if model_n_jobs is None else int(model_n_jobs)
    unique_folds = sorted(np.unique(folds))
    for fold in unique_folds:
        held_out = folds == fold
        training = ~held_out
        if progress is not None:
            progress(
                f"Fold {int(fold) + 1}/{len(unique_folds)}: "
                f"training={int(training.sum())}, "
                f"held_out={int(held_out.sum())}"
            )

        support = GowerReferenceSupport(
            numeric_columns=numeric,
            categorical_columns=categorical,
            n_neighbors=int(support_config["neighbours_per_treatment_arm"]),
            quantile=0.95,
            n_jobs=support_n_jobs,
        ).fit(data.loc[training], treatment[training])
        reference_score[held_out] = support.score(data.loc[held_out])
        reference_supported[held_out] = (
            reference_score[held_out] <= support.threshold_
        ).astype(np.int8)
        support_thresholds[int(fold)] = support.threshold_

        predictions = fit_effect_estimators(
            data.loc[training],
            data.loc[held_out],
            treatment[training],
            outcome[training],
            numeric_columns=numeric,
            categorical_columns=categorical,
            estimator_config=estimator_config,
        )
        causal_forest_effect[held_out] = predictions.causal_forest
        causal_forest_standard_error[held_out] = (
            predictions.causal_forest_standard_error
        )
        penalised_logistic_effect[held_out] = predictions.penalised_logistic
        t_learner_effect[held_out] = predictions.gradient_boosted_t_learner
        outcome_risk_no_aspirin[held_out] = predictions.outcome_risk_no_aspirin
        outcome_risk_aspirin[held_out] = predictions.outcome_risk_aspirin

    data_adequate = data_adequacy_check(
        data, config["gate"]["data_adequacy"]["essential_variables"]
    )
    return assemble_primary_analysis(
        data,
        folds=folds,
        data_adequate=data_adequate,
        reference_supported=reference_supported,
        reference_score=reference_score,
        causal_forest_effect=causal_forest_effect,
        causal_forest_standard_error=causal_forest_standard_error,
        penalised_logistic_effect=penalised_logistic_effect,
        t_learner_effect=t_learner_effect,
        outcome_risk_no_aspirin=outcome_risk_no_aspirin,
        outcome_risk_aspirin=outcome_risk_aspirin,
        support_thresholds=support_thresholds,
        propensity=float(
            config["data"]["aspirin"]["nominal_randomisation_probability"]
        ),
        analysis_row_id=row_id,
    )
