#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import pandas as pd


TRAIN_FRAC = 0.8
VAL_FRAC = 0.1


def assign_split(user_id: str) -> Literal["train", "validation", "test"]:
    """Deterministic user-level split via SHA-1 bucketing (80 / 10 / 10)."""
    digest = int(hashlib.sha1(user_id.encode()).hexdigest(), 16)
    bucket = (digest % 1000) / 1000.0
    if bucket < TRAIN_FRAC:
        return "train"
    if bucket < TRAIN_FRAC + VAL_FRAC:
        return "validation"
    return "test"


LABEL_TYPES: dict[str, str] = {
    "age": "continuous",
    "BiologicalSex": "binary",
    "Diabetes": "binary",
    "Hdl": "continuous",
    "WeightKilograms": "continuous",
    "Hypertension": "binary",
    "Ldl": "continuous",
    "TotalCholesterol": "continuous",
    "WakeUpTime_categories": "ordinal",
    "BMI_categories": "ordinal",
    "BMI_values": "continuous",
    "SystolicBloodPressure": "continuous",
    "feel_worthwhile1": "ordinal",
    "feel_worthwhile2": "ordinal",
    "feel_worthwhile3": "ordinal",
    "feel_worthwhile4": "ordinal",
    "happiness_categories": "ordinal",
    "happiness": "continuous",
    "cardiovascular_disease": "binary",
    "Heart Failure or CHF": "binary",
    "Atrial fibrillation (Afib)": "binary",
    "PH": "binary",
    "CAD": "binary",
    "Congenital Heart": "binary",
    "Peripheral/Systemic Vascular Disease": "binary",
    "Cerebrovascular Disease": "binary",
    "satisfiedwith_life": "ordinal",
    "sleep_diagnosis1": "binary",
    "GoSleepTime_categories": "ordinal",
    "vigorous_act": "continuous",
    "work": "binary",
    "blood_pressure_categories": "ordinal",
    "sleep_time_categories": "ordinal",
}

BINARY_LABELS = sorted(k for k, v in LABEL_TYPES.items() if v == "binary")
ORDINAL_LABELS = sorted(k for k, v in LABEL_TYPES.items() if v == "ordinal")
CONTINUOUS_LABELS = sorted(k for k, v in LABEL_TYPES.items() if v == "continuous")
ALL_LABELS = sorted(LABEL_TYPES.keys())

_META_COLUMNS = frozenset({"user_id", "week_start", "date", "n_valid_days", "n_valid_hours"})


class LabelLookup:
    """Join MHC-benchmark labels to SensorTSLM samples by (user_id, date) or user_id alone.

    The parquet may contain ``week_start`` (weekly) or ``date`` (daily) as the
    timestamp column — or neither for user-level label files.

    Sentinel values (-1 for integer labels, -1.0 for continuous labels)
    are converted to NaN by default.

    Methods:
        join_ondate  – sample-level join keyed by (user_id, date/week_start).
        join_onlabels – user-level join keyed by user_id only (no date required).
    """

    def __init__(self, parquet_path: str | Path) -> None:
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Labels lookup parquet not found: {path}")

        self.df = pd.read_parquet(path)
        self.df["user_id"] = self.df["user_id"].astype(str)

        if "week_start" in self.df.columns:
            self._date_col: str | None = "week_start"
            self.df["week_start"] = self.df["week_start"].astype(str)
        elif "date" in self.df.columns:
            self._date_col = "date"
            self.df["date"] = self.df["date"].astype(str)
        else:
            self._date_col = None  # user-level parquet, no date column

    @property
    def date_column(self) -> str | None:
        return self._date_col

    @property
    def label_names(self) -> list[str]:
        return [col for col in self.df.columns if col not in _META_COLUMNS]

    @property
    def num_samples(self) -> int:
        return len(self.df)

    def join_ondate(
        self,
        user_ids: list[str],
        dates: list[str],
        labels: list[str] | None = None,
    ) -> pd.DataFrame:
        """Sample-level join keyed by (user_id, date/week_start).

        Rows are aligned with the input lists. Missing samples get NaN.
        Requires the parquet to have a ``week_start`` or ``date`` column.
        """
        if self._date_col is None:
            raise ValueError(
                "join_ondate requires a parquet with a 'week_start' or 'date' column. "
                "For user-level parquets, use join_onlabels instead."
            )
        if len(user_ids) != len(dates):
            raise ValueError("user_ids and dates must have the same length")

        label_cols = self.label_names if labels is None else labels
        for name in label_cols:
            if name not in LABEL_TYPES:
                raise ValueError(f"Unknown label: {name!r}")

        keys = pd.DataFrame({"user_id": user_ids, self._date_col: dates})
        keys["user_id"] = keys["user_id"].astype(str)
        keys[self._date_col] = keys[self._date_col].astype(str)

        merged = keys.merge(
            self.df[["user_id", self._date_col] + label_cols],
            on=["user_id", self._date_col],
            how="left",
            validate="many_to_one",
        )

        for col in label_cols:
            label_type = LABEL_TYPES[col]
            sentinel = -1.0 if label_type == "continuous" else -1
            merged[col] = merged[col].replace(sentinel, float("nan"))

        return merged[label_cols]

    def join_onlabels(
        self,
        user_ids: list[str] | None = None,
        labels: list[str] | None = None,
    ) -> pd.DataFrame:
        """User-level join keyed by user_id only (no date required).

        Returns a DataFrame with columns [user_id] + label_cols.
        If user_ids is None, returns labels for all users in the parquet.
        Rows for unknown user_ids get NaN. Sentinel values are replaced with NaN.
        """
        label_cols = self.label_names if labels is None else labels
        for name in label_cols:
            if name not in LABEL_TYPES:
                raise ValueError(f"Unknown label: {name!r}")

        if user_ids is not None:
            keys = pd.DataFrame({"user_id": [str(u) for u in user_ids]})
            result = keys.merge(
                self.df[["user_id"] + label_cols],
                on="user_id",
                how="left",
            )
        else:
            result = self.df[["user_id"] + label_cols].copy()

        for col in label_cols:
            label_type = LABEL_TYPES[col]
            sentinel = -1.0 if label_type == "continuous" else -1
            result[col] = result[col].replace(sentinel, float("nan"))

        return result[["user_id"] + label_cols]
