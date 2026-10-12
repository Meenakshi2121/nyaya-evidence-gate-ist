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

