#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Metabolic prediction evaluation task.

Ground truth : Static MHC survey labels (BiologicalSex, BMI_categories, …).
Input        : One week of hourly Apple Watch sensor data
               (19 channels × 168 h: 7 continuous + 12 binary sleep/workout channels).
Answer format: Model picks one of the provided label options on the last line.
Metrics      : F1, Precision, Recall, Accuracy with bootstrap SEs.

One MetabolicTask instance corresponds to one label column.  All metabolic
tasks share the same prompt structure and parser — only the question text,
the list of valid label values, and the positive class differ.

Supported label columns (binary):
    BiologicalSex        "Male" / "Female"          positive = "Male"
    Diabetes             "True" / "False"           positive = "True"
    Hypertension         "True" / "False"           positive = "True"
    cardiovascular_disease "True" / "False"         positive = "True"

Supported label columns (multi-class, evaluated as binary OvR):
    BMI_categories       "Normal weight" / "Overweight" / "Obesity" / "Underweight"
                         positive = "Overweight"  (most common non-normal class)

Design rationale
----------------
We ask the model to pick exactly one named option from the label set.  This is
equivalent to a pairwise binary choice when |labels| == 2, and extends cleanly
to multi-class without requiring separate task files per label.

LLMs cannot output calibrated probabilities in text form, so this evaluation
reports F1 rather than AUROC. We operate zero-shot and only get a hard label,
so F1 + bootstrap SE is the appropriate fit for this setup.
"""
from __future__ import annotations

import re  # used in parse_prediction

from evaluation.metrics.classification import binary_metrics
from evaluation.tasks.base import EvalSample, EvalTask
from time_series_datasets.mhc_label_lookup import LABEL_TYPES

# ---------------------------------------------------------------------------
# Per-label configuration: question text, valid values, positive class
# ---------------------------------------------------------------------------

_LABEL_CONFIG: dict[str, dict] = {
    "BiologicalSex": {
        "question": "Is this person biologically male or female?",
        "hint": (
            "- Resting and active heart rate (typically lower resting HR in males)\n"
            "- Step count, distance, and active energy levels across the week"
        ),
        "values": ["Male", "Female"],
        "positive_label": "Male",
    },
    "BMI_categories": {
        "question": (
            "Based on activity and heart rate patterns, how would you classify "
            "this person's body weight? Options: Normal weight, Underweight, "
            "Overweight, Obesity."
        ),
        "hint": (
            "- Overall activity levels and step counts\n"
            "- Active energy expenditure during daily activity\n"
            "- Resting heart rate (often elevated with higher BMI)"
        ),
        "values": ["Normal weight", "Underweight", "Overweight", "Obesity"],
        "positive_label": "Overweight",
    },
    "Diabetes": {
        "question": "Does this person have diabetes? Answer True or False.",
        "hint": (
            "- Resting heart rate variability and elevated baseline HR\n"
            "- Step count and overall daily activity levels\n"
            "- Active energy expenditure patterns"
        ),
        "values": ["True", "False"],
        "positive_label": "True",
    },
    "Hypertension": {
        "question": "Does this person have hypertension (high blood pressure)? Answer True or False.",
        "hint": (
            "- Resting heart rate patterns throughout the week\n"
            "- Heart rate recovery after physical activity\n"
            "- Overall activity and sedentary time"
        ),
        "values": ["True", "False"],
        "positive_label": "True",
    },
    "cardiovascular_disease": {
        "question": "Does this person have a cardiovascular disease? Answer True or False.",
        "hint": (
            "- Resting heart rate and heart rate during activity\n"
            "- Exercise tolerance (step count, distance during active periods)\n"
            "- Active energy expenditure relative to activity level"
        ),
        "values": ["True", "False"],
        "positive_label": "True",
    },
}

_PROMPT_TEMPLATE = """\
You are a health informatics researcher analyzing anonymized Apple Watch sensor data.
This is a scientific evaluation benchmark — no real identities are involved.

The plots show one week (168 hours) of wearable sensor recordings from one anonymized participant. \
Each channel shows hourly aggregated values across 7 consecutive days.

Task: {question}

Patterns that may be relevant:
{hint}

Instructions: You must select exactly one answer from the options below. \
Do not refuse or explain uncertainty — simply pick the most likely option based on the data patterns. \
Write your chosen answer as the last line of your response.

Options: {options}\
"""


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

class MetabolicTask(EvalTask):
    """Binary or multi-class metabolic prediction from one week of sensor data.

    Args:
        label_column: MHC survey label column, e.g. "BiologicalSex".
                      Must be a key in _LABEL_CONFIG.
    """

    name = "metabolic"

    def __init__(self, label_column: str, task_name: str) -> None:
        if label_column not in _LABEL_CONFIG:
            raise ValueError(
                f"Unknown label_column '{label_column}'. "
                f"Supported: {sorted(_LABEL_CONFIG)}\n"
                f"All MHC labels (from mhc_label_lookup.LABEL_TYPES): {sorted(LABEL_TYPES)}"
            )
        cfg = _LABEL_CONFIG[label_column]
        self._label_column = label_column
        self._values: list[str] = cfg["values"]
        self._positive_label: str = cfg["positive_label"]
        self._prompt = _PROMPT_TEMPLATE.format(
            question=cfg["question"],
            hint=cfg["hint"],
            options=", ".join(f'"{v}"' for v in cfg["values"]),
        )
        self.name = task_name

    def build_prompt(self, sample: EvalSample) -> str:
        return self._prompt

    # Patterns that indicate the model refused rather than answered.
    _REFUSAL_PATTERNS = re.compile(
        r"\b(i('m| am) sorry|i can'?t|i cannot|unable to|not possible|"
        r"i('m| am) not able|it('s| is) not possible)\b",
        re.IGNORECASE,
    )

    def parse_prediction(self, raw: str) -> str:
        """Extract the first matching label value from model output.

        Checks the last non-empty line first, then falls back to the full text.
        Matching is case-insensitive and whole-word.

        A line that contains a refusal marker is skipped even if it also
        contains a label string (e.g. "I cannot tell if male or female").
        Returns 'unknown' if no valid label is found.
        """
        text = raw.strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        def _match(s: str) -> str | None:
            if self._REFUSAL_PATTERNS.search(s):
                return None
            lower = s.lower()
            for value in self._values:
                if re.search(r"\b" + re.escape(value.lower()) + r"\b", lower):
                    return value
            return None

        for line in reversed(lines):
            hit = _match(line)
            if hit is not None:
                return hit

        # Final fallback: full text, but only if not a refusal
        if not self._REFUSAL_PATTERNS.search(text):
            lower = text.lower()
            for value in self._values:
                if re.search(r"\b" + re.escape(value.lower()) + r"\b", lower):
                    return value

        return "unknown"

    def aggregate_metrics(
        self,
        ground_truths: list[str],
        predictions: list[str],
    ) -> dict[str, float | int]:
        """Compute binary classification metrics against the positive label.

        Unknown predictions are counted separately and excluded from scoring.
        For multi-class labels (e.g. BMI_categories) this computes one-vs-rest
        F1 for the designated positive class.
        """
        n_unknown = sum(1 for p in predictions if p == "unknown")
        filtered = [
            (gt, pred)
            for gt, pred in zip(ground_truths, predictions)
            if pred != "unknown"
        ]

        if not filtered:
            return {
                "f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0,
                "n_samples": len(ground_truths),
                "n_unknown": n_unknown,
                "parse_rate": 0.0,
            }

        gts, preds = zip(*filtered)
        metrics = binary_metrics(
            list(gts), list(preds), positive_label=self._positive_label
        )
        metrics["n_unknown"] = n_unknown
        metrics["parse_rate"] = round(len(filtered) / len(ground_truths), 4)
        return metrics
