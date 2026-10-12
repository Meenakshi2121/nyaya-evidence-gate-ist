import numpy as np
import pytest

from nyaya_gate.gate import build_gate, unanimous_action_agreement
from nyaya_gate.policies import gated_policy, individualised_policy


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

