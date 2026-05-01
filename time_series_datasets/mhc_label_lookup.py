#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from pathlib import Path

import pandas as pd


# =============================================================================
# SHARED — label-type metadata
# =============================================================================
# Used by every task family that touches MHC labels (metabolic via the parquet,
# activity via the workout:* channels, plus future families).

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


# =============================================================================
# METABOLIC — int→string decode tables, per-label config, post-prompt builder
# =============================================================================

# -----------------------------------------------------------------------------
# Authoritative int -> string decoding tables.
# Mirrors upstream MHC-benchmark (~/MHC-benchmark/):
#   - Binary:    src/labels/api.py:126-141 (_to_bool normalizes strings/bools
#                to {0, 1}). All binary labels use {0: "False", 1: "True"}
#                except BiologicalSex, which is sourced from raw "Male"/"Female"
#                strings and maps to {0: "Female", 1: "Male"}.
#   - Ordinal:   data/labels/ordinal_dictionary.json.
# Continuous labels require no decoding (floats pass through).
# Keep in sync if upstream encoding changes.
# -----------------------------------------------------------------------------

_BINARY_OVERRIDES: dict[str, dict[int, str]] = {
    "BiologicalSex": {0: "Female", 1: "Male"},
}
_DEFAULT_BINARY: dict[int, str] = {0: "False", 1: "True"}

_ORDINAL_DECODE: dict[str, dict[int, str]] = {
    "BMI_categories":            {0: "Underweight", 1: "Normal weight", 2: "Overweight", 3: "Obesity", 4: "Morbid Obesity"},
    "WakeUpTime_categories":     {0: "Normal Riser", 1: "Early Riser", 2: "Late Riser", 3: "Very Late Riser"},
    "GoSleepTime_categories":    {0: "Normal Sleeper", 1: "Early Sleeper", 2: "Late Sleeper", 3: "Very Late Sleeper", 4: "Shift Worker"},
    "sleep_time_categories":     {0: "Normal", 1: "Short", 2: "Too Long", 3: "Insufficient"},
    "blood_pressure_categories": {0: "Normal", 1: "Elevated", 2: "Hypertension_Stage_1", 3: "Hypertension_Stage_2", 4: "Hypertensive_Crisis"},
    "feel_worthwhile1":          {0: "Very High", 1: "High", 2: "Medium", 3: "Low"},
    "feel_worthwhile2":          {0: "Very High", 1: "High", 2: "Medium", 3: "Low"},
    "feel_worthwhile3":          {0: "Very High", 1: "High", 2: "Medium", 3: "Low"},
    "feel_worthwhile4":          {0: "Very High", 1: "High", 2: "Medium", 3: "Low"},
    "happiness_categories":      {0: "Very High", 1: "High", 2: "Medium", 3: "Low"},
    "satisfiedwith_life":        {0: "Very High", 1: "High", 2: "Medium", 3: "Low"},
}


def decode_labels(df: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    """Decode int-encoded labels to canonical strings per MHC-benchmark.

    Binary and ordinal columns are mapped to their canonical label strings.
    Continuous columns pass through. NaN entries and values not present in
    the mapping are left untouched (pandas ``replace`` semantics).
    """
    for col in label_cols:
        label_type = LABEL_TYPES[col]
        if label_type == "binary":
            mapping = _BINARY_OVERRIDES.get(col, _DEFAULT_BINARY)
        elif label_type == "ordinal":
            mapping = _ORDINAL_DECODE[col]
        else:
            continue
        df[col] = df[col].replace(mapping)
    return df


# Presentation metadata for the Metabolic task family: the natural-language
# question posed to the model and the set of acceptable answer strings.
METABOLIC_LABEL_CONFIG: dict[str, dict[str, str | list[str]]] = {
    "BiologicalSex": {
        "question": "Is this person biologically male or female?",
        "values": ["Male", "Female"],
    },
    "BMI_categories": {
        "question": (
            "Which BMI category best matches this participant: "
            "Underweight, Normal weight, Overweight, Obesity, or Morbid Obesity?"
        ),
        "values": [
            "Underweight",
            "Normal weight",
            "Overweight",
            "Obesity",
            "Morbid Obesity",
        ],
    },
    "Hypertension": {
        "question": "Does this person have hypertension?",
        "values": ["True", "False"],
    },
    "work": {
        "question": "Does this person engage in regular work activity?",
        "values": ["True", "False"],
    },
    "sleep_diagnosis1": {
        "question": "Does this person have a diagnosed sleep disorder?",
        "values": ["True", "False"],
    },
    "happiness_categories": {
        "question": (
            "How would you rate this person's happiness level: "
            "Very High, High, Medium, or Low?"
        ),
        "values": ["Very High", "High", "Medium", "Low"],
    },
    # Zero-shot eval: not in DEFAULT_METABOLIC_LABELS, the model is not
    # fine-tuned on these. Reported metrics reflect baseline LLM capability.
    "Diabetes": {
        "question": "Does this person have diabetes?",
        "values": ["True", "False"],
    },
    "cardiovascular_disease": {
        "question": "Does this person have cardiovascular disease?",
        "values": ["True", "False"],
    },
}


def get_metabolic_label_config(label_column: str) -> dict[str, str | list[str]]:
    if label_column not in METABOLIC_LABEL_CONFIG:
        raise ValueError(
            f"Unknown metabolic label '{label_column}'. "
            f"Supported: {sorted(METABOLIC_LABEL_CONFIG)}"
        )
    return METABOLIC_LABEL_CONFIG[label_column]


def _wrap_in_rules(question: str, values: list[str]) -> str:
    """Compose a closed-set post_prompt from a question and value list.

    Helper for ``build_metabolic_post_prompt``. If a future task family also
    wants the same single-class "Answer: <label>" framing it can call this
    directly; activity uses a sequence-output format and bypasses this helper.
    """
    values_str = ", ".join(values)
    return (
        f"{question}\n\n"
        f"Possible answers: {values_str}\n\n"
        "Rules:\n"
        "- Base your answer on the sensor patterns above.\n"
        "- You MUST give a classification even if the signal is unclear — "
        "state limitations but still make your best guess.\n"
        "- Never respond with a question.\n"
        "- You MUST respond with ONLY <label>, exactly as written in the listed possible answers."
    )


def build_metabolic_post_prompt(label: str) -> str:
    """Render the post_prompt for ``label`` from its METABOLIC_LABEL_CONFIG entry.

    Shared by ``MHCMetabolicQADataset`` (eval) and ``MHCMultiLabelQADataset``
    (training) so the model sees byte-identical framing in both phases — the
    canonical source of truth lives next to the data it formats.
    """
    cfg = METABOLIC_LABEL_CONFIG.get(label)
    if cfg is None:
        ltype = LABEL_TYPES[label]
        return f"Based on the sensor data, predict the value of {label} ({ltype})."
    return _wrap_in_rules(cfg["question"], cfg["values"])


# =============================================================================
# ACTIVITY — workout-channel labels and post-prompt builder
# =============================================================================

# Channel-name → human-readable workout label. Used at training time to
# derive the dominant-workout class for each sliding window.
# Mirrors mhc.constants.ACTIVITY_CHANNELS in coverage and order.
WORKOUT_LABELS: dict[str, str] = {
    "workout:HKWorkoutActivityTypeWalking": "walking",
    "workout:HKWorkoutActivityTypeCycling": "cycling",
    "workout:HKWorkoutActivityTypeRunning": "running",
    "workout:HKWorkoutActivityTypeOther": "other",
    "workout:HKWorkoutActivityTypeMixedMetabolicCardioTraining": "mixed cardio training",
    "workout:HKWorkoutActivityTypeTraditionalStrengthTraining": "strength training",
    "workout:HKWorkoutActivityTypeElliptical": "elliptical",
    "workout:HKWorkoutActivityTypeHighIntensityIntervalTraining": "high-intensity interval training",
    "workout:HKWorkoutActivityTypeFunctionalStrengthTraining": "functional strength training",
    "workout:HKWorkoutActivityTypeYoga": "yoga",
}

# Closed-set classes for activity classification: the 10 workout types plus
# the explicit "no workout" class (windows with no contiguous active workout).
# Single multi-class head, mirrors the LSM (ICLR 2025) framing.
ACTIVITY_CLASSES: list[str] = list(WORKOUT_LABELS.values()) + ["no workout"]


def build_activity_bin_post_prompt(n_bins: int, bin_size_min: int) -> str:
    """Render the post_prompt for full-recording bin-labelling.

    The model gets the full recording as input and must output one activity
    label per consecutive ``bin_size_min``-minute bin in chronological order
    (sequence-labelling pattern: per-time-step classification given full
    context). The input shape matches the captioning pipeline's full-
    recording shape — only the structured-bin answer format differs.
    """
    classes_str = ", ".join(ACTIVITY_CLASSES)
    total_min = n_bins * bin_size_min
    return (
        f"This recording covers {total_min} consecutive minutes. Output the "
        f"workout activity for each of the {n_bins} consecutive "
        f"{bin_size_min}-minute bins (ordered from start to end).\n\n"
        f"Possible answers per bin: {classes_str}\n\n"
        "Rules:\n"
        f"- Output EXACTLY {n_bins} labels separated by commas.\n"
        "- Use only the listed possible answers, exactly as written.\n"
        "- Base your answer on the sensor patterns above.\n"
        f"- Each label corresponds to one {bin_size_min}-minute bin in chronological order.\n"
        "- Do not generate explanations or any other text."
    )


# =============================================================================
# SHARED — labels parquet loader
# =============================================================================

class LabelLookup:
    """Join MHC-benchmark labels to SensorTSLM samples.

    The parquet may contain ``week_start`` (weekly) or ``date`` (daily) as the
    timestamp column, or neither for user-level label files (one row per user).

    Sentinel values (-1 for integer labels, -1.0 for continuous labels)
    are converted to NaN by default.
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
            self._date_col = None  # user-level parquet

    @property
    def date_column(self) -> str | None:
        return self._date_col

    @property
    def label_names(self) -> list[str]:
        return [col for col in self.df.columns if col not in _META_COLUMNS]

    @property
    def num_samples(self) -> int:
        return len(self.df)

    def join(
        self,
        user_ids: list[str],
        dates: list[str],
        labels: list[str] | None = None,
    ) -> pd.DataFrame:
        """Sample-level join keyed by (user_id, date/week_start).

        If the parquet has no date column (user-level labels), the join
        falls back to user_id only — every sample from the same user
        receives the same labels.

        Rows are aligned with the input lists. Missing samples get NaN.
        """
        if len(user_ids) != len(dates):
            raise ValueError("user_ids and dates must have the same length")

        label_cols = self.label_names if labels is None else labels
        for name in label_cols:
            if name not in LABEL_TYPES:
                raise ValueError(f"Unknown label: {name!r}")

        if self._date_col is not None:
            keys = pd.DataFrame({"user_id": user_ids, self._date_col: dates})
            keys["user_id"] = keys["user_id"].astype(str)
            keys[self._date_col] = keys[self._date_col].astype(str)
            merge_on = ["user_id", self._date_col]
            right_cols = ["user_id", self._date_col] + label_cols
        else:
            keys = pd.DataFrame({"user_id": [str(u) for u in user_ids]})
            merge_on = ["user_id"]
            right_cols = ["user_id"] + label_cols

        merged = keys.merge(
            self.df[right_cols],
            on=merge_on,
            how="left",
            validate="many_to_one",
        )

        for col in label_cols:
            label_type = LABEL_TYPES[col]
            sentinel = -1.0 if label_type == "continuous" else -1
            merged[col] = merged[col].replace(sentinel, float("nan"))

        return merged[label_cols]
