"""Create final percentile confidence intervals after all bootstrap runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyaya_gate.bootstrap import (  # noqa: E402
    atomic_write_json,
    bootstrap_analysis_fingerprint,
    bootstrap_interval_table,
    load_bootstrap_records,
    point_metrics_from_summary,
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
        "--bootstrap-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "bootstrap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    target = int(config["inference"]["bootstrap_replicates"])
    confidence_level = float(config["inference"]["confidence_level"])
    fingerprint = bootstrap_analysis_fingerprint(
        PROJECT_ROOT,
        config_path=args.config,
        input_path=args.input,
        point_summary_path=args.point_summary,
    )
    records = load_bootstrap_records(
        args.bootstrap_dir / "replicates",
        analysis_fingerprint=fingerprint,
    )
    found = {int(record["replicate"]) for record in records}
    expected = set(range(target))
    if found != expected:
        missing = sorted(expected - found)
        preview = ", ".join(str(value) for value in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(
            f"Cannot summarise: {len(records)}/{target} replicates are complete. "
            f"Missing replicate indices: {preview}{suffix}"
        )

    with args.point_summary.open("r", encoding="utf-8") as stream:
        primary_summary = json.load(stream)
    point_estimates = point_metrics_from_summary(primary_summary)
    intervals = bootstrap_interval_table(
        records,
        point_estimates,
        confidence_level=confidence_level,
    )

    table_path = PROJECT_ROOT / "results" / "tables" / "bootstrap_confidence_intervals.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(table_path, index=False)
    summary = {
        "analysis_fingerprint": fingerprint,
        "bootstrap_replicates": target,
        "confidence_level": confidence_level,
        "interval_method": "percentile",
        "whole_pipeline_repeated": True,
        "estimands": intervals.to_dict(orient="records"),
        "inference_status": "complete",
    }
    atomic_write_json(PROJECT_ROOT / "results" / "bootstrap_summary.json", summary)
    print(intervals.to_string(index=False))
    print(
        "Bootstrap inference complete. Aggregate confidence-interval outputs "
        "are ready for review; replicate files remain excluded from Git.",
        flush=True,
    )


if __name__ == "__main__":
    main()
