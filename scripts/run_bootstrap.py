"""Run resumable batches of the prespecified whole-pipeline bootstrap."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyaya_gate.bootstrap import (  # noqa: E402
    BOOTSTRAP_SCHEMA_VERSION,
    atomic_write_json,
    bootstrap_analysis_fingerprint,
    load_bootstrap_records,
    replicate_path,
    run_bootstrap_replicate,
)
from nyaya_gate.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "analysis.yml",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ist_analysis.csv",
    )
    parser.add_argument(
        "--point-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "primary_analysis_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "bootstrap",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First zero-indexed replicate to run (inclusive).",
    )
    parser.add_argument(
        "--stop",
        type=int,
        help="Final zero-indexed boundary (exclusive); defaults to start + 1.",
    )
    parser.add_argument(
        "--model-jobs",
        type=int,
        default=-1,
        help="Parallel jobs used inside support and causal-forest models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    target = int(config["inference"]["bootstrap_replicates"])
    stop = args.start + 1 if args.stop is None else args.stop
    if args.start < 0 or stop <= args.start or stop > target:
        raise ValueError(
            f"Requested range [{args.start}, {stop}) must lie within [0, {target})."
        )
    for required in (args.input, args.point_summary):
        if not required.exists():
            raise FileNotFoundError(f"Required analysis input not found: {required}")

    fingerprint = bootstrap_analysis_fingerprint(
        PROJECT_ROOT,
        config_path=args.config,
        input_path=args.input,
        point_summary_path=args.point_summary,
    )
    replicate_dir = args.output_dir / "replicates"
    plan_path = args.output_dir / "bootstrap_plan.json"
    plan = {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "analysis_fingerprint": fingerprint,
        "target_replicates": target,
        "replicate_indexing": "zero_based_start_inclusive_stop_exclusive",
        "resampling": "within_joint_aspirin_outcome_strata",
        "master_seed": int(config["project"]["random_seed"]),
    }
    if plan_path.exists():
        with plan_path.open("r", encoding="utf-8") as stream:
            existing_plan = json.load(stream)
        if existing_plan != plan:
            raise ValueError(
                "The existing bootstrap directory belongs to a different "
                "analysis. Move it aside before starting a new run."
            )
    else:
        atomic_write_json(plan_path, plan)

    existing_records = load_bootstrap_records(
        replicate_dir,
        analysis_fingerprint=fingerprint,
    )
    completed = {int(record["replicate"]) for record in existing_records}
    data = pd.read_csv(args.input, low_memory=False)

    for replicate in range(args.start, stop):
        output_path = replicate_path(replicate_dir, replicate)
        if replicate in completed:
            print(
                f"Bootstrap replicate {replicate} already complete; skipping.",
                flush=True,
            )
            continue
        print(
            f"Starting bootstrap replicate {replicate} "
            f"({replicate + 1} of {target}).",
            flush=True,
        )
        started = time.perf_counter()
        record = run_bootstrap_replicate(
            data,
            config,
            replicate=replicate,
            analysis_fingerprint=fingerprint,
            model_n_jobs=args.model_jobs,
            progress=lambda message: print(message, flush=True),
        )
        record["elapsed_seconds"] = float(time.perf_counter() - started)
        atomic_write_json(output_path, record)
        completed.add(replicate)
        print(
            f"Completed bootstrap replicate {replicate} in "
            f"{record['elapsed_seconds'] / 60:.1f} minutes; saved {output_path}.",
            flush=True,
        )

    progress_payload = {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "analysis_fingerprint": fingerprint,
        "completed_replicates": len(completed),
        "target_replicates": target,
        "next_missing_replicate": next(
            (index for index in range(target) if index not in completed), None
        ),
    }
    atomic_write_json(args.output_dir / "progress.json", progress_payload)
    print(
        f"Bootstrap progress: {len(completed)}/{target} replicates complete.",
        flush=True,
    )


if __name__ == "__main__":
    main()
