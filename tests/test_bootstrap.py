import json

import numpy as np
import pandas as pd
import pytest

from nyaya_gate.bootstrap import (
    PRIMARY_ESTIMANDS,
    atomic_write_json,
    bootstrap_interval_table,
    bootstrap_seeds,
    load_bootstrap_records,
    point_metrics_from_summary,
    replicate_path,
    stratified_bootstrap_sample,
)


def test_stratified_bootstrap_preserves_joint_stratum_counts():
    data = pd.DataFrame(
        {
            "study_row_id": np.arange(40),
            "aspirin": np.repeat([0, 0, 1, 1], 10),
            "outcome": np.tile(np.repeat([0, 1], 10), 2),
        }
    )
    sample = stratified_bootstrap_sample(data, random_state=123)

    expected = data.groupby(["aspirin", "outcome"]).size()
    observed = sample.groupby(["aspirin", "outcome"]).size()
    pd.testing.assert_series_equal(observed, expected)
    np.testing.assert_array_equal(sample["bootstrap_row_id"], np.arange(40))
    assert sample["bootstrap_row_id"].is_unique


def test_bootstrap_seeds_are_reproducible_and_replicate_specific():
    first = bootstrap_seeds(20261012, 7)
    repeated = bootstrap_seeds(20261012, 7)
    next_replicate = bootstrap_seeds(20261012, 8)

    assert first == repeated
    assert first != next_replicate
    assert len(set(first.values())) == 3


def test_point_metrics_and_percentile_interval_table():
    summary = {
        "policy_expected_risks": {
            "treat_all_aspirin": 0.60,
            "ungated_individualisation": 0.62,
            "evidence_gated": 0.61,
            "uncertainty_only_matched_coverage": 0.615,
        },
        "within_group_gain": {"accepted": 0.01, "deferred": -0.02},
    }
    point = point_metrics_from_summary(summary)
    assert point == pytest.approx(
        {
            "gated_minus_treat_all": 0.01,
            "gated_minus_ungated_individualisation": -0.01,
            "gated_minus_uncertainty_only_matched_coverage": -0.005,
            "accepted_minus_deferred_gain": 0.03,
        }
    )

    records = []
    for offset in (-0.02, -0.01, 0.0, 0.01, 0.02):
        records.append(
            {
                "metrics": {
                    estimand: point[estimand] + offset
                    for estimand in PRIMARY_ESTIMANDS
                }
            }
        )
    intervals = bootstrap_interval_table(
        records,
        point,
        confidence_level=0.80,
    )
    assert list(intervals["estimand"]) == list(PRIMARY_ESTIMANDS)
    assert (intervals["bootstrap_replicates"] == 5).all()
    np.testing.assert_allclose(intervals["bootstrap_mean"], list(point.values()))


def test_atomic_records_are_loaded_in_replicate_order(tmp_path):
    replicate_dir = tmp_path / "replicates"
    for replicate in (2, 0, 1):
        atomic_write_json(
            replicate_path(replicate_dir, replicate),
            {
                "schema_version": 1,
                "status": "complete",
                "replicate": replicate,
                "analysis_fingerprint": "same-analysis",
                "metrics": {},
            },
        )

    records = load_bootstrap_records(
        replicate_dir,
        analysis_fingerprint="same-analysis",
    )
    assert [record["replicate"] for record in records] == [0, 1, 2]
    with replicate_path(replicate_dir, 1).open("r", encoding="utf-8") as stream:
        assert json.load(stream)["status"] == "complete"
