"""Prepare the IST data for modelling without fitting outcome models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyaya_gate.audit import read_csv_robust  # noqa: E402
from nyaya_gate.config import load_config  # noqa: E402


BASELINE_COLUMNS = [
    "RDELAY",
    "RCONSC",
    "SEX",
    "AGE",
    "RSLEEP",
    "RATRIAL",
    "RCT",
    "RVISINF",
    "RHEP24",
    "RASP3",
    "RSBP",
    "RDEF1",
    "RDEF2",
    "RDEF3",
    "RDEF4",
    "RDEF5",
    "RDEF6",
    "RDEF7",
    "RDEF8",
    "STYPE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "analysis.yml",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_codes(series: pd.Series, expected: set[str], name: str) -> None:
    observed = set(series.dropna().astype("string").str.strip().unique())
    unexpected = observed - expected
    if unexpected:
        raise ValueError(f"Unexpected {name} codes: {sorted(unexpected)}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    raw_path = PROJECT_ROOT / data_config["raw_file"]
    raw = read_csv_robust(raw_path)

    columns = data_config["columns"]
    hospital_column = columns["hospital_id"]
    aspirin_column = columns["aspirin_allocation"]
    heparin_column = columns["heparin_allocation"]
    outcome_column = columns["six_month_outcome"]
    required = {
        hospital_column,
        aspirin_column,
        heparin_column,
        outcome_column,
        *BASELINE_COLUMNS,
    }
    missing_columns = sorted(required - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Required IST columns are missing: {missing_columns}")

    prepared = raw.loc[:, sorted(required)].copy()
    text_columns = prepared.select_dtypes(include=["object", "string"]).columns
    prepared[text_columns] = prepared[text_columns].apply(
        lambda series: series.astype("string").str.strip().replace("", pd.NA)
    )

    check_codes(prepared[aspirin_column], {"Y", "N"}, "aspirin")
    check_codes(prepared[heparin_column], {"N", "L", "M", "H"}, "heparin")
    outcome_config = data_config["outcome"]
    adverse_codes = {int(code) for code in outcome_config["adverse_codes"]}
    favourable_codes = {int(code) for code in outcome_config["favourable_codes"]}
    missing_codes = {int(code) for code in outcome_config["missing_codes"]}
    expected_outcome_codes = adverse_codes | favourable_codes | missing_codes
    check_codes(
        prepared[outcome_column],
        {str(code) for code in expected_outcome_codes},
        "outcome",
    )

    prepared.insert(0, "study_row_id", range(1, len(prepared) + 1))
    prepared["hospital_id"] = prepared[hospital_column]
    prepared["aspirin"] = prepared[aspirin_column].map({"Y": 1, "N": 0}).astype("Int64")
    prepared["heparin_group"] = prepared[heparin_column].map(
        {"N": "none", "L": "low", "M": "higher", "H": "higher"}
    )

    outcome_numeric = pd.to_numeric(prepared[outcome_column], errors="raise")
    outcome_map = {code: 1 for code in adverse_codes}
    outcome_map.update({code: 0 for code in favourable_codes})
    outcome_map.update({code: pd.NA for code in missing_codes})
    prepared["outcome"] = outcome_numeric.map(outcome_map).astype("Int64")

    analysis = prepared.loc[prepared["outcome"].notna()].copy()
    if len(raw) != 19_435 or len(analysis) != 19_285:
        raise ValueError(
            "Unexpected IST sample size: "
            f"raw={len(raw)}, analysable={len(analysis)}; expected 19435 and 19285."
        )

    processed_dir = PROJECT_ROOT / "data" / "processed"
    results_dir = PROJECT_ROOT / "results"
    table_dir = results_dir / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    prepared.to_csv(processed_dir / "ist_full_prepared.csv", index=False)
    analysis.to_csv(processed_dir / "ist_analysis.csv", index=False)

    missingness = (
        prepared[BASELINE_COLUMNS]
        .isna()
        .sum()
        .rename("n_missing")
        .to_frame()
        .assign(missing_percent=lambda frame: frame["n_missing"] / len(prepared) * 100)
        .rename_axis("variable")
        .reset_index()
        .sort_values(["n_missing", "variable"], ascending=[False, True])
    )
    missingness.to_csv(table_dir / "analysis_variable_missingness.csv", index=False)

    summary = {
        "source_file": str(raw_path.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256(raw_path),
        "raw_rows": int(len(raw)),
        "analysis_rows": int(len(analysis)),
        "excluded_missing_outcome": int(prepared["outcome"].isna().sum()),
        "aspirin_allocation": {
            str(key): int(value)
            for key, value in analysis["aspirin"].value_counts().sort_index().items()
        },
        "six_month_outcome": {
            "death_or_dependency": int((analysis["outcome"] == 1).sum()),
            "alive_and_independent": int((analysis["outcome"] == 0).sum()),
        },
        "heparin_group": {
            str(key): int(value)
            for key, value in analysis["heparin_group"].value_counts().items()
        },
        "note": "Patient-level prepared files are excluded from version control.",
    }
    with (results_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
