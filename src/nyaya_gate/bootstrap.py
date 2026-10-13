"""Deterministic, resumable whole-pipeline bootstrap utilities."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .analysis import PrimaryAnalysis
from .pipeline import ProgressCallback, run_cross_fitted_analysis

BOOTSTRAP_SCHEMA_VERSION = 1
PRIMARY_ESTIMANDS = (
    "gated_minus_treat_all",
    "gated_minus_ungated_individualisation",
    "gated_minus_uncertainty_only_matched_coverage",
    "accepted_minus_deferred_gain",
)

ESTIMAND_METADATA = {
    "gated_minus_treat_all": {
        "description": "Expected-risk difference: gated minus treat-all aspirin",
        "favourable_direction": "below_zero",
    },
    "gated_minus_ungated_individualisation": {
        "description": "Expected-risk difference: gated minus ungated individualisation",
        "favourable_direction": "below_zero",
    },
    "gated_minus_uncertainty_only_matched_coverage": {
        "description": "Expected-risk difference: gated minus uncertainty-only at matched coverage",
        "favourable_direction": "below_zero",
    },
    "accepted_minus_deferred_gain": {
        "description": "Gain from individualisation: accepted minus deferred",
        "favourable_direction": "above_zero",
    },
}


def bootstrap_seeds(master_seed: int, replicate: int) -> dict[str, int]:
    """Derive independent reproducible seeds for one replicate."""
    if master_seed < 0 or replicate < 0:
        raise ValueError("master_seed and replicate must be non-negative.")
    sequence = np.random.SeedSequence([int(master_seed), int(replicate)])
    children = sequence.spawn(3)
    names = ("resample", "folds", "estimators")
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(names, children, strict=True)
    }


def stratified_bootstrap_sample(
    data: pd.DataFrame,
    *,
    random_state: int,
    treatment_column: str = "aspirin",
    outcome_column: str = "outcome",
) -> pd.DataFrame:
    """Resample with replacement within joint treatment-outcome strata."""
    if "bootstrap_row_id" in data.columns:
        raise ValueError("Input data already contain bootstrap_row_id.")
    missing = sorted({treatment_column, outcome_column} - set(data.columns))
    if missing:
        raise ValueError(f"Bootstrap stratification columns are missing: {missing}")
    treatment = data[treatment_column].to_numpy(dtype=np.int8)
    outcome = data[outcome_column].to_numpy(dtype=np.int8)
    if not np.isin(treatment, [0, 1]).all() or not np.isin(
        outcome, [0, 1]
    ).all():
        raise ValueError("Bootstrap treatment and outcome must be binary.")

    strata = 2 * treatment + outcome
    rng = np.random.default_rng(random_state)
    draws: list[np.ndarray] = []
    for stratum in range(4):
        rows = np.flatnonzero(strata == stratum)
        if len(rows) == 0:
            raise ValueError(f"Bootstrap stratum {stratum} is empty.")
        draws.append(rng.choice(rows, size=len(rows), replace=True))
    sampled_rows = np.concatenate(draws)
    rng.shuffle(sampled_rows)

    sample = data.iloc[sampled_rows].reset_index(drop=True).copy()
    sample["bootstrap_row_id"] = np.arange(len(sample), dtype=np.int64)
    return sample


def analysis_metrics(analysis: PrimaryAnalysis) -> dict[str, float]:
    """Extract all scalar metrics retained for each bootstrap replicate."""
    risks = analysis.policy_risks.set_index("policy")["expected_risk"].to_dict()
    gains = analysis.group_gains.set_index("gate_group")[
        "gain_from_individualisation"
    ].to_dict()
    group_rows = analysis.group_gains.set_index("gate_group")
    return {
        "gate_coverage": float(analysis.summary["gate"]["coverage"]),
        "treat_all_risk": float(risks["treat_all_aspirin"]),
        "ungated_individualisation_risk": float(
            risks["ungated_individualisation"]
        ),
        "evidence_gated_risk": float(risks["evidence_gated"]),
        "uncertainty_only_matched_coverage_risk": float(
            risks["uncertainty_only_matched_coverage"]
        ),
        "gated_minus_treat_all": float(
            risks["evidence_gated"] - risks["treat_all_aspirin"]
        ),
        "gated_minus_ungated_individualisation": float(
            risks["evidence_gated"] - risks["ungated_individualisation"]
        ),
        "gated_minus_uncertainty_only_matched_coverage": float(
            risks["evidence_gated"]
            - risks["uncertainty_only_matched_coverage"]
        ),
        "accepted_gain": float(gains["accepted"]),
        "deferred_gain": float(gains["deferred"]),
        "accepted_minus_deferred_gain": float(
            gains["accepted"] - gains["deferred"]
        ),
        "accepted_no_aspirin_proportion": float(
            group_rows.loc["accepted", "proportion_recommended_no_aspirin"]
        ),
        "deferred_no_aspirin_proportion": float(
            group_rows.loc["deferred", "proportion_recommended_no_aspirin"]
        ),
    }


def point_metrics_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    """Recover the four inferential point estimates from the primary summary."""
    risks = summary["policy_expected_risks"]
    gains = summary["within_group_gain"]
    return {
        "gated_minus_treat_all": float(
            risks["evidence_gated"] - risks["treat_all_aspirin"]
        ),
        "gated_minus_ungated_individualisation": float(
            risks["evidence_gated"] - risks["ungated_individualisation"]
        ),
        "gated_minus_uncertainty_only_matched_coverage": float(
            risks["evidence_gated"]
            - risks["uncertainty_only_matched_coverage"]
        ),
        "accepted_minus_deferred_gain": float(
            gains["accepted"] - gains["deferred"]
        ),
    }


def run_bootstrap_replicate(
    data: pd.DataFrame,
    config: dict[str, Any],
    *,
    replicate: int,
    analysis_fingerprint: str,
    model_n_jobs: int = -1,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Repeat resampling, cross-fitting, gate construction and evaluation."""
    if "study_row_id" not in data.columns or not data["study_row_id"].is_unique:
        raise ValueError("Source study_row_id values must be present and unique.")
    master_seed = int(config["project"]["random_seed"])
    seeds = bootstrap_seeds(master_seed, replicate)
    sample = stratified_bootstrap_sample(
        data,
        random_state=seeds["resample"],
    )

    def report(message: str) -> None:
        if progress is not None:
            progress(f"Bootstrap replicate {replicate}: {message}")

    analysis = run_cross_fitted_analysis(
        sample,
        config,
        fold_seed=seeds["folds"],
        estimator_seed=seeds["estimators"],
        analysis_row_id=sample["bootstrap_row_id"].to_numpy(),
        model_n_jobs=model_n_jobs,
        progress=report,
    )
    stratum_counts = (
        sample.groupby(["aspirin", "outcome"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    return {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "status": "complete",
        "replicate": int(replicate),
        "analysis_fingerprint": analysis_fingerprint,
        "seeds": seeds,
        "sample_rows": int(len(sample)),
        "unique_source_rows": int(sample["study_row_id"].nunique()),
        "stratum_counts": [
            {
                "aspirin": int(row.aspirin),
                "outcome": int(row.outcome),
                "n": int(row.n),
            }
            for row in stratum_counts.itertuples(index=False)
        ],
        "metrics": analysis_metrics(analysis),
    }


def bootstrap_interval_table(
    records: Iterable[dict[str, Any]],
    point_estimates: dict[str, float],
    *,
    confidence_level: float,
) -> pd.DataFrame:
    """Construct percentile intervals for the prespecified estimands."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    materialised = list(records)
    if len(materialised) < 2:
        raise ValueError("At least two bootstrap replicates are required.")
    alpha = 1.0 - confidence_level
    rows: list[dict[str, Any]] = []
    for estimand in PRIMARY_ESTIMANDS:
        if estimand not in point_estimates:
            raise ValueError(f"Point estimate is missing for {estimand}.")
        values = np.asarray(
            [record["metrics"][estimand] for record in materialised],
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite bootstrap value found for {estimand}.")
        metadata = ESTIMAND_METADATA[estimand]
        rows.append(
            {
                "estimand": estimand,
                "description": metadata["description"],
                "favourable_direction": metadata["favourable_direction"],
                "point_estimate": float(point_estimates[estimand]),
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_standard_error": float(np.std(values, ddof=1)),
                "ci_lower": float(np.quantile(values, alpha / 2)),
                "ci_upper": float(np.quantile(values, 1 - alpha / 2)),
                "confidence_level": float(confidence_level),
                "interval_method": "percentile",
                "bootstrap_replicates": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def fingerprint_files(paths: Iterable[Path], *, root: Path) -> str:
    """Hash analysis inputs and code so incompatible batches cannot mix."""
    root = root.resolve()
    unique_paths = sorted({Path(path).resolve() for path in paths})
    digest = hashlib.sha256()
    for path in unique_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Fingerprint input not found: {path}")
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = path.name
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def bootstrap_analysis_fingerprint(
    project_root: Path,
    *,
    config_path: Path,
    input_path: Path,
    point_summary_path: Path,
) -> str:
    """Fingerprint every input that determines bootstrap estimates."""
    root = project_root.resolve()
    paths = [
        config_path,
        input_path,
        point_summary_path,
        root / "pyproject.toml",
        root / "requirements.txt",
        *(root / "src" / "nyaya_gate").glob("*.py"),
    ]
    return fingerprint_files(paths, root=root)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON by atomic replacement so interruption cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def load_bootstrap_records(
    replicate_dir: Path,
    *,
    analysis_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    """Load, validate and order completed replicate records."""
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in sorted(replicate_dir.glob("replicate_*.json")):
        with path.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
        if record.get("schema_version") != BOOTSTRAP_SCHEMA_VERSION:
            raise ValueError(f"Unsupported bootstrap schema in {path}.")
        if record.get("status") != "complete":
            raise ValueError(f"Incomplete bootstrap record: {path}.")
        if analysis_fingerprint is not None and record.get(
            "analysis_fingerprint"
        ) != analysis_fingerprint:
            raise ValueError(
                f"Bootstrap record {path} belongs to a different analysis."
            )
        replicate = int(record["replicate"])
        if replicate < 0 or path.name != replicate_path(
            replicate_dir, replicate
        ).name:
            raise ValueError(f"Bootstrap filename and replicate disagree: {path}.")
        if replicate in seen:
            raise ValueError(f"Duplicate bootstrap replicate {replicate}.")
        seen.add(replicate)
        records.append(record)
    return sorted(records, key=lambda record: int(record["replicate"]))


def replicate_path(replicate_dir: Path, replicate: int) -> Path:
    """Return the stable filename for one zero-indexed replicate."""
    if replicate < 0:
        raise ValueError("replicate must be non-negative.")
    return replicate_dir / f"replicate_{replicate:04d}.json"
