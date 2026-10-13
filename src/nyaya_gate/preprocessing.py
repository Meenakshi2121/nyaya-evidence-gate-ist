"""Training-fold-only preprocessing for the treatment-effect estimators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class FoldPreprocessor:
    """Median/mode imputation, one-hot encoding and missingness indicators.

    All learned quantities come from the training fold. The transform returns a
    dense numeric matrix shared by the three prespecified causal estimators.
    """

    def __init__(
        self,
        numeric_columns: Sequence[str],
        categorical_columns: Sequence[str],
    ) -> None:
        self.numeric_columns = tuple(numeric_columns)
        self.categorical_columns = tuple(categorical_columns)
        if not self.numeric_columns and not self.categorical_columns:
            raise ValueError("At least one model variable is required.")
        overlap = set(self.numeric_columns) & set(self.categorical_columns)
        if overlap:
            raise ValueError(f"Variables cannot be both numeric and categorical: {overlap}")

    def fit(self, data: pd.DataFrame) -> Self:
        self._check_columns(data)
        numeric = self._numeric(data)
        self.numeric_medians_ = numeric.median().fillna(0.0)

        categorical = self._categorical(data)
        self.categorical_modes_: dict[str, str] = {}
        for name in self.categorical_columns:
            observed = categorical[name].dropna()
            mode = str(observed.mode().iloc[0]) if not observed.empty else "__MISSING__"
            self.categorical_modes_[name] = mode

        self.one_hot_ = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
        if self.categorical_columns:
            self.one_hot_.fit(self._fill_categorical(categorical))
        self.feature_names_out_ = self._make_feature_names()
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        self._check_columns(data)
        blocks: list[np.ndarray] = []
        missing_blocks: list[np.ndarray] = []

        if self.numeric_columns:
            numeric = self._numeric(data)
            missing_blocks.append(numeric.isna().to_numpy(dtype=float))
            blocks.append(
                numeric.fillna(self.numeric_medians_).to_numpy(dtype=np.float64)
            )

        if self.categorical_columns:
            categorical = self._categorical(data)
            missing_blocks.append(categorical.isna().to_numpy(dtype=float))
            blocks.append(
                self.one_hot_.transform(self._fill_categorical(categorical))
            )

        blocks.extend(missing_blocks)
        transformed = np.concatenate(blocks, axis=1)
        if not np.isfinite(transformed).all():
            raise ValueError("Preprocessing produced non-finite model inputs.")
        return transformed

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        return self.fit(data).transform(data)

    def get_feature_names_out(self) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)

    def _check_columns(self, data: pd.DataFrame) -> None:
        required = set(self.numeric_columns) | set(self.categorical_columns)
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"Model variables are missing: {missing}")

    def _numeric(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.loc[:, self.numeric_columns].apply(pd.to_numeric, errors="coerce")

    def _categorical(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def _make_feature_names(self) -> list[str]:
        names = list(self.numeric_columns)
        if self.categorical_columns:
            names.extend(self.one_hot_.get_feature_names_out(self.categorical_columns))
        names.extend(f"{name}__missing" for name in self.numeric_columns)
        names.extend(f"{name}__missing" for name in self.categorical_columns)
        return [str(name) for name in names]
