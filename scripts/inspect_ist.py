"""Run the first-stage reproducible audit of the downloaded IST CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyaya_gate.audit import run_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "IST_corrected.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run_audit(arguments.input, arguments.output_dir)
    print(json.dumps(result, indent=2))

