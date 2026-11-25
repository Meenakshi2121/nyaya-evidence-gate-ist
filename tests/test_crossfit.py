import numpy as np

from nyaya_gate.crossfit import (
    make_grouped_stratified_folds,
    make_stratified_folds,
)


def test_joint_stratification_and_row_order_invariance():
    treatment = np.tile([0, 0, 1, 1], 10)
    outcome = np.tile([0, 1, 0, 1], 10)
    identifiers = np.arange(100, 140)
    first = make_stratified_folds(
        treatment,
        outcome,
        n_splits=5,
        random_state=7,
        participant_id=identifiers,
    )

    permutation = np.random.default_rng(8).permutation(len(treatment))
    second_permuted = make_stratified_folds(
        treatment[permutation],
        outcome[permutation],
        n_splits=5,
        random_state=7,
        participant_id=identifiers[permutation],
    )
    second = np.empty_like(second_permuted)
    second[permutation] = second_permuted
    np.testing.assert_array_equal(first, second)

    for fold in range(5):
        rows = first == fold
        strata = 2 * treatment[rows] + outcome[rows]
        np.testing.assert_array_equal(np.bincount(strata, minlength=4), [2, 2, 2, 2])


def test_grouped_stratification_is_group_safe_and_row_order_invariant():
    group_id = np.repeat(np.arange(100, 140), 2)
    group_treatment = np.tile([0, 0, 1, 1], 10)
    group_outcome = np.tile([0, 1, 0, 1], 10)
    treatment = np.repeat(group_treatment, 2)
    outcome = np.repeat(group_outcome, 2)

    first = make_grouped_stratified_folds(
        treatment,
        outcome,
        n_splits=5,
        random_state=7,
        group_id=group_id,
    )

    for identifier in np.unique(group_id):
        assert np.unique(first[group_id == identifier]).size == 1

    permutation = np.random.default_rng(8).permutation(len(treatment))
    second_permuted = make_grouped_stratified_folds(
        treatment[permutation],
        outcome[permutation],
        n_splits=5,
        random_state=7,
        group_id=group_id[permutation],
    )
    second = np.empty_like(second_permuted)
    second[permutation] = second_permuted
    np.testing.assert_array_equal(first, second)

    strata = 2 * treatment + outcome
    for fold in range(5):
        assert set(strata[first == fold]) == {0, 1, 2, 3}
