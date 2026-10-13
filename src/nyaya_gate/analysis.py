"""Assembly and reporting of held-out evidence-gate results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import aipw_policy_scores
from .gate import build_gate, unanimous_action_agreement
from .policies import (
    gated_policy,
    individualised_policy,
    matched_uncertainty_acceptance,
)


@dataclass(frozen=True)
class PrimaryAnalysis:
    """Aggregate and patient-level outputs from one cross-fitted partition."""

    patient_results: pd.DataFrame
    policy_risks: pd.DataFrame
    group_gains: pd.DataFrame
    check_patterns: pd.DataFrame
    selection_overlap: pd.DataFrame
    summary: dict[str, Any]


def assemble_primary_analysis(
    data: pd.DataFrame,
    *,
    folds: np.ndarray,
    data_adequate: np.ndarray,
    reference_supported: np.ndarray,
    reference_score: np.ndarray,
    causal_forest_effect: np.ndarray,
    causal_forest_standard_error: np.ndarray,
    penalised_logistic_effect: np.ndarray,
    t_learner_effect: np.ndarray,
    outcome_risk_no_aspirin: np.ndarray,
    outcome_risk_aspirin: np.ndarray,
    support_thresholds: dict[int, float],
    propensity: float = 0.5,
    analysis_row_id: np.ndarray | None = None,
) -> PrimaryAnalysis:
    """Construct policies and point estimates from strictly held-out values."""
    required = {"study_row_id", "aspirin", "outcome"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Analysis columns are missing: {missing}")

    arrays = {
        "fold": np.asarray(folds),
        "data_adequate": np.asarray(data_adequate, dtype=np.int8),
        "reference_supported": np.asarray(reference_supported, dtype=np.int8),
        "reference_score": np.asarray(reference_score, dtype=float),
        "causal_forest_effect": np.asarray(causal_forest_effect, dtype=float),
        "causal_forest_standard_error": np.asarray(
            causal_forest_standard_error, dtype=float
        ),
        "penalised_logistic_effect": np.asarray(
            penalised_logistic_effect, dtype=float
        ),
        "t_learner_effect": np.asarray(t_learner_effect, dtype=float),
        "outcome_risk_no_aspirin": np.asarray(
            outcome_risk_no_aspirin, dtype=float
        ),
        "outcome_risk_aspirin": np.asarray(outcome_risk_aspirin, dtype=float),
    }
    lengths = {len(data), *(len(values) for values in arrays.values())}
    if len(lengths) != 1:
        raise ValueError("All held-out result arrays must match the analysis data.")
    if not np.isin(arrays["data_adequate"], [0, 1]).all():
        raise ValueError("data_adequate must be binary.")
    if not np.isin(arrays["reference_supported"], [0, 1]).all():
        raise ValueError("reference_supported must be binary.")

    if analysis_row_id is None:
        row_id = data["study_row_id"].to_numpy()
    else:
        row_id = np.asarray(analysis_row_id)
    if row_id.ndim != 1 or len(row_id) != len(data):
        raise ValueError("analysis_row_id must contain one value per row.")
    if len(np.unique(row_id)) != len(row_id):
        raise ValueError("analysis_row_id values must be unique.")

    effects = np.column_stack(
        [
            arrays["causal_forest_effect"],
            arrays["penalised_logistic_effect"],
            arrays["t_learner_effect"],
        ]
    )
    estimator_agreement = unanimous_action_agreement(effects)
    gate = build_gate(
        arrays["data_adequate"],
        arrays["reference_supported"],
        estimator_agreement,
    )
    individualised = individualised_policy(arrays["causal_forest_effect"])
    treat_all = np.ones(len(data), dtype=np.int8)
    gated = gated_policy(gate, individualised, default_treatment=1)
    uncertainty_acceptance = matched_uncertainty_acceptance(
        gate,
        arrays["causal_forest_standard_error"],
        arrays["fold"],
        row_id,
    )
    uncertainty_policy = gated_policy(
        uncertainty_acceptance, individualised, default_treatment=1
    )

    lower = arrays["causal_forest_effect"] - 1.96 * arrays[
        "causal_forest_standard_error"
    ]
    upper = arrays["causal_forest_effect"] + 1.96 * arrays[
        "causal_forest_standard_error"
    ]
    interval_acceptance = (
        np.isfinite(lower) & np.isfinite(upper) & ((upper < 0) | (lower > 0))
    ).astype(np.int8)
    interval_policy = gated_policy(
        interval_acceptance, individualised, default_treatment=1
    )

    y = data["outcome"].to_numpy(dtype=np.int8)
    a = data["aspirin"].to_numpy(dtype=np.int8)
    policies = {
        "treat_all_aspirin": treat_all,
        "ungated_individualisation": individualised,
        "evidence_gated": gated,
        "uncertainty_only_matched_coverage": uncertainty_policy,
        "uncertainty_interval_sensitivity": interval_policy,
    }
    scores = {
        name: aipw_policy_scores(
            y,
            a,
            policy,
            arrays["outcome_risk_no_aspirin"],
            arrays["outcome_risk_aspirin"],
            propensity,
        )
        for name, policy in policies.items()
    }
    default_scores = scores["treat_all_aspirin"]
    individualised_scores = scores["ungated_individualisation"]

    policy_risks = pd.DataFrame(
        {
            "policy": list(policies),
            "expected_risk": [float(np.mean(scores[name])) for name in policies],
            "n_recommended_aspirin": [
                int(policies[name].sum()) for name in policies
            ],
            "n_recommended_no_aspirin": [
                int((policies[name] == 0).sum()) for name in policies
            ],
        }
    )
    default_risk = float(
        policy_risks.loc[
            policy_risks["policy"] == "treat_all_aspirin", "expected_risk"
        ].iloc[0]
    )
    policy_risks["risk_difference_vs_treat_all"] = (
        policy_risks["expected_risk"] - default_risk
    )

    group_rows: list[dict[str, Any]] = []
    for label, mask in (("accepted", gate == 1), ("deferred", gate == 0)):
        if not mask.any():
            raise ValueError(f"The gate produced no {label} patients.")
        treat_all_risk = float(np.mean(default_scores[mask]))
        individualised_risk = float(np.mean(individualised_scores[mask]))
        no_aspirin = int(((individualised == 0) & mask).sum())
        group_rows.append(
            {
                "gate_group": label,
                "n": int(mask.sum()),
                "n_recommended_no_aspirin": no_aspirin,
                "proportion_recommended_no_aspirin": float(
                    no_aspirin / mask.sum()
                ),
                "treat_all_risk": treat_all_risk,
                "individualised_risk": individualised_risk,
                "gain_from_individualisation": treat_all_risk
                - individualised_risk,
            }
        )
    group_gains = pd.DataFrame(group_rows)
    gains = group_gains.set_index("gate_group")["gain_from_individualisation"]
    principal_contrast = float(gains["accepted"] - gains["deferred"])

    patterns = pd.DataFrame(
        {
            "data_adequate": arrays["data_adequate"],
            "reference_supported": arrays["reference_supported"],
            "estimators_agree": estimator_agreement,
        }
    )
    check_patterns = (
        patterns.value_counts(sort=False)
        .rename("n")
        .reset_index()
        .sort_values(
            ["data_adequate", "reference_supported", "estimators_agree"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    check_patterns["proportion"] = check_patterns["n"] / len(data)

    overlap_categories = {
        "accepted_by_both": (gate == 1) & (uncertainty_acceptance == 1),
        "gate_only": (gate == 1) & (uncertainty_acceptance == 0),
        "uncertainty_only": (gate == 0) & (uncertainty_acceptance == 1),
        "deferred_by_both": (gate == 0) & (uncertainty_acceptance == 0),
    }
    selection_overlap = pd.DataFrame(
        {
            "selection_pattern": list(overlap_categories),
            "n": [int(mask.sum()) for mask in overlap_categories.values()],
        }
    )
    selection_overlap["proportion"] = selection_overlap["n"] / len(data)
    accepted_intersection = int(overlap_categories["accepted_by_both"].sum())
    accepted_union = int(((gate == 1) | (uncertainty_acceptance == 1)).sum())
    acceptance_agreement = float((gate == uncertainty_acceptance).mean())
    accepted_set_jaccard = float(accepted_intersection / accepted_union)

    patient_results = pd.DataFrame(
        {
            "study_row_id": data["study_row_id"].to_numpy(),
            "analysis_row_id": row_id,
            "fold": arrays["fold"],
            "aspirin": a,
            "outcome": y,
            "data_adequate": arrays["data_adequate"],
            "reference_score": arrays["reference_score"],
            "reference_supported": arrays["reference_supported"],
            "causal_forest_effect": arrays["causal_forest_effect"],
            "causal_forest_standard_error": arrays[
                "causal_forest_standard_error"
            ],
            "penalised_logistic_effect": arrays["penalised_logistic_effect"],
            "t_learner_effect": arrays["t_learner_effect"],
            "estimators_agree": estimator_agreement,
            "gate_accepted": gate,
            "uncertainty_accepted": uncertainty_acceptance,
            "interval_accepted": interval_acceptance,
            "individualised_policy": individualised,
            "gated_policy": gated,
            "uncertainty_policy": uncertainty_policy,
            "outcome_risk_no_aspirin": arrays["outcome_risk_no_aspirin"],
            "outcome_risk_aspirin": arrays["outcome_risk_aspirin"],
        }
    )
    for name, values in scores.items():
        patient_results[f"aipw_score_{name}"] = values

    summary: dict[str, Any] = {
        "analysis_rows": int(len(data)),
        "folds": int(len(np.unique(arrays["fold"]))),
        "gate": {
            "accepted_n": int(gate.sum()),
            "deferred_n": int((gate == 0).sum()),
            "coverage": float(gate.mean()),
            "check_failures": {
                "data_adequacy": int((arrays["data_adequate"] == 0).sum()),
                "reference_support": int(
                    (arrays["reference_supported"] == 0).sum()
                ),
                "estimator_agreement": int((estimator_agreement == 0).sum()),
            },
        },
        "matched_uncertainty_coverage": float(uncertainty_acceptance.mean()),
        "gate_vs_uncertainty_selection": {
            "accepted_by_both_n": accepted_intersection,
            "gate_only_n": int(overlap_categories["gate_only"].sum()),
            "uncertainty_only_n": int(
                overlap_categories["uncertainty_only"].sum()
            ),
            "deferred_by_both_n": int(
                overlap_categories["deferred_by_both"].sum()
            ),
            "acceptance_agreement": acceptance_agreement,
            "accepted_set_jaccard": accepted_set_jaccard,
        },
        "interval_sensitivity_coverage": float(interval_acceptance.mean()),
        "support_threshold_by_fold": {
            str(key): float(value)
            for key, value in sorted(support_thresholds.items())
        },
        "policy_expected_risks": {
            str(row.policy): float(row.expected_risk)
            for row in policy_risks.itertuples(index=False)
        },
        "policy_risk_differences_vs_treat_all": {
            str(row.policy): float(row.risk_difference_vs_treat_all)
            for row in policy_risks.itertuples(index=False)
        },
        "within_group_gain": {
            str(row.gate_group): float(row.gain_from_individualisation)
            for row in group_gains.itertuples(index=False)
        },
        "principal_contrast_accepted_minus_deferred_gain": principal_contrast,
        "inference_status": "point_estimates_only_pending_whole_pipeline_bootstrap",
    }
    return PrimaryAnalysis(
        patient_results=patient_results,
        policy_risks=policy_risks,
        group_gains=group_gains,
        check_patterns=check_patterns,
        selection_overlap=selection_overlap,
        summary=summary,
    )
