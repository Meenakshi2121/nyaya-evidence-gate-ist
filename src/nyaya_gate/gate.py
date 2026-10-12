"""Non-compensatory evidence-gate construction."""

from __future__ import annotations

import numpy as np


def build_gate(
    data_adequate: np.ndarray,
    reference_supported: np.ndarray,
    estimators_agree: np.ndarray,
) -> np.ndarray:
    """Return one only when all three essential checks pass."""
    checks = [
        np.asarray(data_adequate, dtype=bool),
        np.asarray(reference_supported, dtype=bool),
        np.asarray(estimators_agree, dtype=bool),
    ]
    lengths = {len(check) for check in checks}
    if len(lengths) != 1:
        raise ValueError("All gate checks must have the same length.")
    return np.logical_and.reduce(checks).astype(np.int8)


def unanimous_action_agreement(effect_estimates: np.ndarray) -> np.ndarray:
    """Check whether all prespecified estimators recommend the same action.

    Negative risk differences recommend aspirin; zero or positive differences
    recommend no aspirin. Rows containing non-finite estimates fail.
    """
    effects = np.asarray(effect_estimates, dtype=float)
    if effects.ndim != 2 or effects.shape[1] < 2:
        raise ValueError("Provide an n-by-m matrix for at least two estimators.")
    finite = np.isfinite(effects).all(axis=1)
    actions = effects < 0
    agreement = np.all(actions == actions[:, [0]], axis=1)
    return (finite & agreement).astype(np.int8)

