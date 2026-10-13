"""Training-fold-only Gower nearest-neighbour reference support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder


@dataclass
class _GowerEncoder:
    """Encode mixed data so Manhattan distance equals Gower distance."""

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]

    def fit(self, data: pd.DataFrame) -> Self:
        self._check_columns(data)
        numeric = data.loc[:, self.numeric_columns].apply(pd.to_numeric, errors="coerce")
        self.numeric_medians_ = numeric.median().fillna(0.0)
        filled_numeric = numeric.fillna(self.numeric_medians_)
        ranges = filled_numeric.max() - filled_numeric.min()
        self.numeric_ranges_ = ranges.mask(ranges == 0, 1.0)
        self.numeric_minima_ = filled_numeric.min()

        categorical = self._normalise_categorical(data)
        self.categorical_modes_: dict[str, str] = {}
        categories: list[list[str]] = []
        for name in self.categorical_columns:
            observed = categorical[name].dropna()
            mode = str(observed.mode().iloc[0]) if not observed.empty else "__MISSING__"
            self.categorical_modes_[name] = mode
            levels = sorted(set(observed.astype(str)) | {mode, "__UNKNOWN__"})
            categories.append(levels)
        self.one_hot_ = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
        self.one_hot_.fit(self._fill_categorical(categorical))
        self.n_original_features_ = len(self.numeric_columns) + len(
            self.categorical_columns
        )
        if self.n_original_features_ == 0:
            raise ValueError("At least one support variable is required.")
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        self._check_columns(data)
        blocks: list[np.ndarray] = []
        weight = 1.0 / self.n_original_features_
        if self.numeric_columns:
            numeric = data.loc[:, self.numeric_columns].apply(
                pd.to_numeric, errors="coerce"
            )
            numeric = numeric.fillna(self.numeric_medians_)
            scaled = (numeric - self.numeric_minima_) / self.numeric_ranges_
            blocks.append(scaled.to_numpy(dtype=float) * weight)
        if self.categorical_columns:
            categorical = self._fill_categorical(self._normalise_categorical(data))
            known = [set(levels) for levels in self.one_hot_.categories_]
            for position, name in enumerate(self.categorical_columns):
                categorical[name] = categorical[name].where(
                    categorical[name].isin(known[position]), "__UNKNOWN__"
                )
            encoded = self.one_hot_.transform(categorical)
            blocks.append(encoded * (0.5 * weight))
        return np.concatenate(blocks, axis=1)

    def _check_columns(self, data: pd.DataFrame) -> None:
        required = set(self.numeric_columns) | set(self.categorical_columns)
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"Support variables are missing: {missing}")

    def _normalise_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.loc[:, self.categorical_columns].copy()
        for name in self.categorical_columns:
            result[name] = result[name].astype("string").str.strip().str.upper()
            result[name] = result[name].replace("", pd.NA)
        return result

    def _fill_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for name, mode in self.categorical_modes_.items():
            result[name] = result[name].fillna(mode).astype(str)
        return result


class GowerReferenceSupport:
    """Evaluate support in both treatment arms using Gower k-nearest neighbours."""

    def __init__(
        self,
        *,
        numeric_columns: tuple[str, ...],
        categorical_columns: tuple[str, ...],
        n_neighbors: int = 50,
        quantile: float = 0.95,
        n_jobs: int | None = -1,
    ) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive.")
        if not 0 < quantile < 1:
            raise ValueError("quantile must lie strictly between zero and one.")
        self.encoder = _GowerEncoder(numeric_columns, categorical_columns)
        self.n_neighbors = n_neighbors
        self.quantile = quantile
        self.n_jobs = n_jobs

    def fit(self, data: pd.DataFrame, treatment: np.ndarray) -> Self:
        a = np.asarray(treatment, dtype=np.int8)
        if len(data) != len(a) or not np.isin(a, [0, 1]).all():
            raise ValueError("Treatment must be a matching binary vector.")
        self.encoder.fit(data)
        transformed = self.encoder.transform(data)
        self.models_: dict[int, NearestNeighbors] = {}
        self.arm_rows_: dict[int, np.ndarray] = {}
        for arm in (0, 1):
            rows = np.flatnonzero(a == arm)
            if len(rows) <= self.n_neighbors:
                raise ValueError(
                    f"Treatment arm {arm} needs more than {self.n_neighbors} rows."
                )
            model = NearestNeighbors(metric="manhattan", algorithm="brute", n_jobs=self.n_jobs)
            model.fit(transformed[rows])
            self.models_[arm] = model
            self.arm_rows_[arm] = rows

        training_scores = self._score_transformed(
            transformed,
            training_row_numbers=np.arange(len(data)),
        )
        self.threshold_ = float(np.quantile(training_scores, self.quantile))
        self.training_scores_ = training_scores
        return self

    def score(self, data: pd.DataFrame) -> np.ndarray:
        """Return the worse arm-specific mean neighbour distance."""
        return self._score_transformed(self.encoder.transform(data))

    def check(self, data: pd.DataFrame) -> np.ndarray:
        """Return one when the support score is within the fitted threshold."""
        return (self.score(data) <= self.threshold_).astype(np.int8)

    def _score_transformed(
        self,
        transformed: np.ndarray,
        training_row_numbers: np.ndarray | None = None,
    ) -> np.ndarray:
        arm_means: list[np.ndarray] = []
        for arm in (0, 1):
            extra = 1 if training_row_numbers is not None else 0
            distances, neighbours = self.models_[arm].kneighbors(
                transformed,
                n_neighbors=self.n_neighbors + extra,
                return_distance=True,
            )
            if training_row_numbers is None:
                arm_means.append(distances.mean(axis=1))
                continue

            global_neighbours = self.arm_rows_[arm][neighbours]
            means = np.empty(len(transformed), dtype=float)
            for row in range(len(transformed)):
                keep = global_neighbours[row] != training_row_numbers[row]
                selected = distances[row, keep][: self.n_neighbors]
                if len(selected) != self.n_neighbors:
                    raise RuntimeError("Could not construct leave-one-out neighbours.")
                means[row] = selected.mean()
            arm_means.append(means)
        return np.maximum(arm_means[0], arm_means[1])
