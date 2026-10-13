"""Observed-data adequacy checks based on the official IST dictionary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


IST_ALLOWED_CODES: dict[str, frozenset[str]] = {
    "RCONSC": frozenset({"F", "D", "U"}),
    "SEX": frozenset({"M", "F"}),
    "RSLEEP": frozenset({"Y", "N"}),
    "RATRIAL": frozenset({"Y", "N"}),
    "RCT": frozenset({"Y", "N"}),
    "RVISINF": frozenset({"Y", "N"}),
    "RHEP24": frozenset({"Y", "N"}),
    "RASP3": frozenset({"Y", "N"}),
    **{f"RDEF{number}": frozenset({"Y", "N", "C"}) for number in range(1, 9)},
    "STYPE": frozenset({"TACS", "PACS", "POCS", "LACS", "OTH"}),
}


def data_adequacy_check(
    data: pd.DataFrame,
    essential_variables: Sequence[str],
    *,
    allowed_codes: Mapping[str, frozenset[str]] = IST_ALLOWED_CODES,
) -> np.ndarray:
    """Return one for rows with observed, dictionary-valid essential data.

    The IST dictionary specifies code sets for categorical variables but does
    not prescribe clinical cut-offs for the numeric variables. Numeric values
    therefore pass when they are present and finite; no post-hoc plausibility
    bounds are invented.
    """
    variables = list(essential_variables)
    missing_columns = sorted(set(variables) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Essential variables are missing: {missing_columns}")

    adequate = np.ones(len(data), dtype=bool)
    for name in variables:
        series = data[name]
        if name in allowed_codes:
            normalised = series.astype("string").str.strip().str.upper()
            adequate &= normalised.notna().to_numpy()
            adequate &= normalised.isin(allowed_codes[name]).to_numpy()
        else:
            numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            adequate &= np.isfinite(numeric)
    return adequate.astype(np.int8)
