"""Deterministic cross-fitting utilities."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def make_stratified_folds(
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
    participant_id: np.ndarray | None = None,
) -> np.ndarray:
    """Assign folds stratified jointly by binary treatment and outcome.

    When participant identifiers are supplied, assignment is invariant to the
    input row order. This matters because the generated IST row identifier is
    also the prespecified deterministic tie-breaker.
    """
    a = np.asarray(treatment)
    y = np.asarray(outcome)
    if a.ndim != 1 or y.ndim != 1 or len(a) != len(y):
        raise ValueError("Treatment and outcome must be equal-length vectors.")
    if not np.isin(a, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Treatment and outcome must be binary.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two.")

    if participant_id is None:
        order = np.arange(len(a))
    else:
        identifiers = np.asarray(participant_id)
        if identifiers.ndim != 1 or len(identifiers) != len(a):
            raise ValueError("participant_id must match treatment and outcome.")
        if len(np.unique(identifiers)) != len(identifiers):
            raise ValueError("participant_id values must be unique.")
        order = np.argsort(identifiers, kind="stable")

    strata = (2 * a.astype(np.int8) + y.astype(np.int8))[order]
    counts = np.bincount(strata, minlength=4)
    if np.any(counts < n_splits):
        raise ValueError("Every treatment-outcome stratum needs n_splits cases.")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    ordered_folds = np.empty(len(a), dtype=np.int16)
    placeholder = np.zeros(len(a), dtype=np.int8)
    for fold, (_, held_out) in enumerate(splitter.split(placeholder, strata)):
        ordered_folds[held_out] = fold

    folds = np.empty(len(a), dtype=np.int16)
    folds[order] = ordered_folds
    return folds


def make_grouped_stratified_folds(
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
    group_id: np.ndarray,
) -> np.ndarray:
    """Assign stratified folds while keeping repeated group rows together."""
    a = np.asarray(treatment)
    y = np.asarray(outcome)
    identifiers = np.asarray(group_id)
    if a.ndim != 1 or y.ndim != 1 or len(a) != len(y):
        raise ValueError("Treatment and outcome must be equal-length vectors.")
    if not np.isin(a, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Treatment and outcome must be binary.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two.")
    if identifiers.ndim != 1 or len(identifiers) != len(a):
        raise ValueError("group_id must match treatment and outcome.")

    strata = 2 * a.astype(np.int8) + y.astype(np.int8)
    order = np.argsort(identifiers, kind="stable")
    ordered_groups = identifiers[order]
    ordered_strata = strata[order]
    group_starts = np.flatnonzero(
        np.r_[True, ordered_groups[1:] != ordered_groups[:-1]]
    )
    group_stops = np.r_[group_starts[1:], len(ordered_groups)]
    group_strata = np.empty(len(group_starts), dtype=np.int8)
    for index, (start, stop) in enumerate(zip(group_starts, group_stops)):
        values = ordered_strata[start:stop]
        if len(values) == 0 or np.any(values != values[0]):
            raise ValueError(
                "Each group_id must belong to one treatment-outcome stratum."
            )
        group_strata[index] = values[0]

    counts = np.bincount(group_strata, minlength=4)
    if np.any(counts < n_splits):
        raise ValueError(
            "Every treatment-outcome stratum needs n_splits source groups."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    ordered_folds = np.empty(len(a), dtype=np.int16)
    placeholder = np.zeros(len(a), dtype=np.int8)
    for fold, (_, held_out) in enumerate(
        splitter.split(placeholder, ordered_strata, groups=ordered_groups)
    ):
        ordered_folds[held_out] = fold

    folds = np.empty(len(a), dtype=np.int16)
    folds[order] = ordered_folds
    return folds
