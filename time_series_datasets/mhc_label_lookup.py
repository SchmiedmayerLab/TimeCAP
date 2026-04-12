#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from pathlib import Path

import pandas as pd


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
    """Join MHC-benchmark labels to SensorTSLM samples by (user_id, date).

    The parquet uses ``week_start`` (weekly) or ``date`` (daily) as the
    timestamp column. This class auto-detects which is present.

    Sentinel values (-1 for integer labels, -1.0 for continuous labels)
    are converted to NaN by default.
    """

    def __init__(self, parquet_path: str | Path) -> None:
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Labels lookup parquet not found: {path}")

        self.df = pd.read_parquet(path)

        if "week_start" in self.df.columns:
            self._date_col = "week_start"
        elif "date" in self.df.columns:
            self._date_col = "date"
        else:
            raise ValueError("Parquet must contain a 'week_start' or 'date' column")

        self.df["user_id"] = self.df["user_id"].astype(str)
        self.df[self._date_col] = self.df[self._date_col].astype(str)

    @property
    def date_column(self) -> str:
        return self._date_col

    @property
    def label_names(self) -> list[str]:
        return [col for col in self.df.columns if col not in _META_COLUMNS]

    @property
    def num_samples(self) -> int:
        return len(self.df)

    def join(self, user_ids: list[str], dates: list[str], labels: list[str] | None = None) -> pd.DataFrame:
        """Batch join: return a DataFrame of label values for a list of samples.

        Rows are aligned with the input lists. Missing samples get NaN.
        Sentinel values are replaced with NaN.
        """
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
