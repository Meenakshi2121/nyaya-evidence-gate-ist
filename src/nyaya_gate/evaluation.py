"""Cross-fitted augmented inverse-probability-weighted policy evaluation."""

from __future__ import annotations

import numpy as np


def aipw_policy_scores(
    outcome: np.ndarray,
    treatment: np.ndarray,
    policy: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: np.ndarray | float,
) -> np.ndarray:
    """Return participant-level AIPW scores for a binary treatment policy."""
    y = np.asarray(outcome, dtype=float)
    a = np.asarray(treatment, dtype=np.int8)
    pi = np.asarray(policy, dtype=np.int8)
    m0 = np.asarray(mu0, dtype=float)
    m1 = np.asarray(mu1, dtype=float)
    p = np.broadcast_to(np.asarray(propensity, dtype=float), y.shape)

    arrays = (y, a, pi, m0, m1, p)
    if len({len(x) for x in arrays}) != 1:
        raise ValueError("All policy-evaluation inputs must have equal length.")
    if not np.isin(a, [0, 1]).all() or not np.isin(pi, [0, 1]).all():
        raise ValueError("Treatment and policy must be binary.")
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("Propensities must lie strictly between zero and one.")

    selected_mu = np.where(pi == 1, m1, m0)
    observed_mu = np.where(a == 1, m1, m0)
    match_probability = np.where(pi == 1, p, 1 - p)
    correction = (a == pi) * (y - observed_mu) / match_probability
    return selected_mu + correction


def aipw_policy_risk(
    outcome: np.ndarray,
    treatment: np.ndarray,
    policy: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: np.ndarray | float,
) -> float:
    """Estimate expected adverse-outcome risk; lower is better."""
    scores = aipw_policy_scores(outcome, treatment, policy, mu0, mu1, propensity)
    return float(np.mean(scores))


def group_policy_gain(
    individualised_scores: np.ndarray,
    treat_all_scores: np.ndarray,
    group: np.ndarray,
) -> float:
    """Estimate risk reduction from individualisation within a gate group.

    Positive values mean lower estimated risk under individualisation.
    """
    ind = np.asarray(individualised_scores, dtype=float)
    default = np.asarray(treat_all_scores, dtype=float)
    mask = np.asarray(group, dtype=bool)
    if not (len(ind) == len(default) == len(mask)):
        raise ValueError("Scores and group indicator must have equal length.")
    if not mask.any():
        raise ValueError("The requested group contains no participants.")
    return float(np.mean(default[mask] - ind[mask]))

