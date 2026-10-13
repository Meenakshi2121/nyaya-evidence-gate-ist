import numpy as np
import pytest

from nyaya_gate.gate import build_gate, unanimous_action_agreement
from nyaya_gate.policies import (
    gated_policy,
    individualised_policy,
    matched_uncertainty_acceptance,
)


def test_gate_is_non_compensatory():
    gate = build_gate(
        np.array([1, 1, 1, 0]),
        np.array([1, 0, 1, 1]),
        np.array([1, 1, 0, 1]),
    )
    np.testing.assert_array_equal(gate, np.array([1, 0, 0, 0]))


def test_unanimous_action_agreement():
    effects = np.array([[-0.03, -0.01, -0.02], [-0.03, 0.01, -0.02]])
    np.testing.assert_array_equal(unanimous_action_agreement(effects), [1, 0])


def test_gated_policy_uses_aspirin_default():
    individualised = individualised_policy(np.array([0.02, 0.03, -0.01]))
    result = gated_policy(np.array([1, 0, 1]), individualised, default_treatment=1)
    np.testing.assert_array_equal(result, np.array([0, 1, 1]))


def test_gate_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        build_gate(np.array([1]), np.array([1, 0]), np.array([1]))


def test_uncertainty_comparator_matches_coverage_within_fold():
    gate = np.array([1, 0, 1, 0, 0, 1])
    uncertainty = np.array([0.3, 0.1, 0.2, 0.4, 0.2, 0.1])
    folds = np.array([0, 0, 0, 1, 1, 1])
    identifiers = np.array([1, 2, 3, 4, 5, 6])
    accepted = matched_uncertainty_acceptance(
        gate, uncertainty, folds, identifiers
    )
    np.testing.assert_array_equal(accepted, [0, 1, 1, 0, 0, 1])
    assert accepted[folds == 0].sum() == gate[folds == 0].sum()
    assert accepted[folds == 1].sum() == gate[folds == 1].sum()

