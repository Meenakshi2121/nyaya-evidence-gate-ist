"""Treatment-policy definitions."""

from __future__ import annotations

import numpy as np


def individualised_policy(effect: np.ndarray) -> np.ndarray:
    """Recommend aspirin when its predicted risk difference is negative."""
    values = np.asarray(effect, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Treatment-effect estimates must be finite.")
    return (values < 0).astype(np.int8)


def gated_policy(
    gate: np.ndarray,
    individualised: np.ndarray,
    default_treatment: int = 1,
) -> np.ndarray:
    """Apply individualisation when accepted and the default when deferred."""
    accepted = np.asarray(gate, dtype=bool)
    policy = np.asarray(individualised, dtype=np.int8)
    if len(accepted) != len(policy):
        raise ValueError("Gate and policy arrays must have the same length.")
    if default_treatment not in (0, 1):
        raise ValueError("default_treatment must be zero or one.")
    return np.where(accepted, policy, default_treatment).astype(np.int8)


def matched_uncertainty_acceptance(
    gate: np.ndarray,
    uncertainty: np.ndarray,
    fold: np.ndarray,
    participant_id: np.ndarray,
) -> np.ndarray:
    """Accept the least-uncertain cases at the gate's coverage in each fold."""
    gate_values = np.asarray(gate, dtype=np.int8)
    uncertainty_values = np.asarray(uncertainty, dtype=float)
    folds = np.asarray(fold)
    identifiers = np.asarray(participant_id)
    if len({len(gate_values), len(uncertainty_values), len(folds), len(identifiers)}) != 1:
        raise ValueError("Matched-coverage inputs must have equal length.")
    if not np.isin(gate_values, [0, 1]).all():
        raise ValueError("gate must be binary.")
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("participant_id values must be unique.")

    accepted = np.zeros(len(gate_values), dtype=np.int8)
    sortable_uncertainty = np.where(np.isfinite(uncertainty_values), uncertainty_values, np.inf)
    for fold_value in np.unique(folds):
        rows = np.flatnonzero(folds == fold_value)
        target = int(gate_values[rows].sum())
        order = np.lexsort((identifiers[rows], sortable_uncertainty[rows]))
        accepted[rows[order[:target]]] = 1
    return accepted

