import numpy as np
import pandas as pd

from nyaya_gate.analysis import assemble_primary_analysis


def test_primary_analysis_assembles_gate_policies_and_matched_coverage():
    n = 12
    data = pd.DataFrame(
        {
            "study_row_id": np.arange(1, n + 1),
            "aspirin": np.tile([0, 1], n // 2),
            "outcome": np.tile([0, 1, 1], n // 3),
        }
    )
    folds = np.repeat([0, 1], n // 2)
    data_adequate = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1])
    reference_supported = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
    forest = np.array([-0.2, -0.1, 0.1, 0.2, -0.2, 0.1] * 2)
    logistic = np.array([-0.1, -0.2, 0.2, 0.1, -0.1, 0.2] * 2)
    t_learner = np.array([-0.3, -0.1, 0.1, 0.2, 0.1, 0.1] * 2)

    result = assemble_primary_analysis(
        data,
        folds=folds,
        data_adequate=data_adequate,
        reference_supported=reference_supported,
        reference_score=np.linspace(0.1, 0.4, n),
        causal_forest_effect=forest,
        causal_forest_standard_error=np.linspace(0.01, 0.12, n),
        penalised_logistic_effect=logistic,
        t_learner_effect=t_learner,
        outcome_risk_no_aspirin=np.full(n, 0.55),
        outcome_risk_aspirin=np.full(n, 0.50),
        support_thresholds={0: 0.3, 1: 0.31},
    )

    patient = result.patient_results
    assert set(patient["gate_accepted"].unique()) <= {0, 1}
    for fold in (0, 1):
        rows = patient["fold"] == fold
        assert patient.loc[rows, "gate_accepted"].sum() == patient.loc[
            rows, "uncertainty_accepted"
        ].sum()
    assert set(result.policy_risks["policy"]) == {
        "treat_all_aspirin",
        "ungated_individualisation",
        "evidence_gated",
        "uncertainty_only_matched_coverage",
        "uncertainty_interval_sensitivity",
    }
    assert set(result.group_gains["gate_group"]) == {"accepted", "deferred"}
    assert result.selection_overlap["n"].sum() == n
    assert set(result.selection_overlap["selection_pattern"]) == {
        "accepted_by_both",
        "gate_only",
        "uncertainty_only",
        "deferred_by_both",
    }
    overlap = result.summary["gate_vs_uncertainty_selection"]
    assert overlap["gate_only_n"] == overlap["uncertainty_only_n"]
    assert 0 <= overlap["acceptance_agreement"] <= 1
    assert 0 <= overlap["accepted_set_jaccard"] <= 1
    assert np.isfinite(result.policy_risks["expected_risk"]).all()
    assert (
        result.summary["gate"]["accepted_n"]
        + result.summary["gate"]["deferred_n"]
        == n
    )
