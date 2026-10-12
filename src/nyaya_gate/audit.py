"""Dataset inspection without assuming undocumented IST codings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_csv_robust(path: str | Path) -> pd.DataFrame:
    """Read a CSV while tolerating common legacy encodings."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. See data/README.md for instructions."
        )
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(csv_path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f"Could not decode {csv_path}") from last_error


def column_inventory(data: pd.DataFrame) -> pd.DataFrame:
    """Return column types, completeness and uniqueness."""
    rows: list[dict[str, Any]] = []
    for name in data.columns:
        series = data[name]
        rows.append(
            {
                "column": str(name),
                "dtype": str(series.dtype),
                "n_non_missing": int(series.notna().sum()),
                "n_missing": int(series.isna().sum()),
                "missing_percent": float(series.isna().mean() * 100),
                "n_unique_non_missing": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values("column", kind="stable")


def value_inventory(data: pd.DataFrame, max_levels: int = 30) -> pd.DataFrame:
    """Count values for categorical or low-cardinality columns."""
    frames: list[pd.DataFrame] = []
    for name in data.columns:
        series = data[name]
        if series.nunique(dropna=False) <= max_levels:
            counts = series.astype("string").fillna("<MISSING>").value_counts(dropna=False)
            frame = counts.rename_axis("value").reset_index(name="count")
            frame.insert(0, "column", str(name))
            frame["percent"] = frame["count"] / len(data) * 100
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["column", "value", "count", "percent"])
    return pd.concat(frames, ignore_index=True)


def run_audit(input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create aggregate inventories and return an audit summary."""
    data = read_csv_robust(input_path)
    out = Path(output_dir)
    table_dir = out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    columns = column_inventory(data)
    values = value_inventory(data)
    missingness = columns[["column", "n_missing", "missing_percent"]].sort_values(
        "missing_percent", ascending=False, kind="stable"
    )

    columns.to_csv(table_dir / "column_inventory.csv", index=False)
    values.to_csv(table_dir / "value_inventory.csv", index=False)
    missingness.to_csv(table_dir / "missingness.csv", index=False)

    summary = {
        "input_file": str(Path(input_path)),
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "duplicate_rows": int(data.duplicated().sum()),
        "fully_complete_columns": int((data.isna().sum() == 0).sum()),
    }
    with (out / "data_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary

