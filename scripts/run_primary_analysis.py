"""Run the prespecified five-fold IST evidence-gate point analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyaya_gate.config import load_config  # noqa: E402
from nyaya_gate.pipeline import run_cross_fitted_analysis  # noqa: E402


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if not args.input.exists():
        raise FileNotFoundError(
            f"Prepared analysis file not found: {args.input}. "
            "Run `python scripts/prepare_ist.py` first."
        )
    data = pd.read_csv(args.input, low_memory=False)

    analysis = run_cross_fitted_analysis(
        data,
        config,
        progress=lambda message: print(message, flush=True),
    )

    processed_dir = PROJECT_ROOT / "data" / "processed"
    results_dir = PROJECT_ROOT / "results"
    table_dir = results_dir / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    analysis.patient_results.to_csv(
        processed_dir / "primary_analysis_patient_level.csv", index=False
    )
    analysis.policy_risks.to_csv(table_dir / "primary_policy_risks.csv", index=False)
    analysis.group_gains.to_csv(table_dir / "primary_group_gains.csv", index=False)
    analysis.check_patterns.to_csv(table_dir / "gate_check_patterns.csv", index=False)
    analysis.selection_overlap.to_csv(
        table_dir / "gate_vs_uncertainty_overlap.csv", index=False
    )
    with (results_dir / "primary_analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(analysis.summary, stream, indent=2)

    print(json.dumps(analysis.summary, indent=2))
    print(
        "Point analysis complete. Patient-level output remains excluded from Git; "
        "aggregate outputs are ready for review.",
        flush=True,
    )


if __name__ == "__main__":
    main()
