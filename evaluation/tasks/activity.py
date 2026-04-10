#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Activity recognition evaluation task.

Ground truth: workout:* binary channels from Apple Watch / HealthKit.
Question:     "Did this person perform a <workout_type> workout?"
Answer:       "yes" or "no"
Metrics:      F1, Precision, Recall, Accuracy with bootstrap standard errors.
"""
from __future__ import annotations

import re

from evaluation.metrics.classification import binary_metrics
from evaluation.tasks.base import EvalSample, EvalTask

_PROMPT_TEMPLATE = """\
You are analyzing Apple Watch sensor data for activity recognition.

The plots show {window_size} minutes of wearable sensor recordings from one person. \
Each channel is plotted as a separate time series.

Question: Did this person perform a {workout_label} workout during this time window?

Patterns to look for:
- Sustained elevated heart rate above resting level
- Consistent non-zero step count or distance
- Increased active energy expenditure

Answer with exactly one word on the last line of your response: "yes" or "no".\
"""


class ActivityTask(EvalTask):
    """Binary yes/no activity recognition task over MHC sensor windows."""

    name = "activity"

    def build_prompt(self, sample: EvalSample) -> str:
        workout_label = sample.metadata.get("workout_label", "workout")
        window_size = sample.metadata.get("window_size", 60)
        return _PROMPT_TEMPLATE.format(
            workout_label=workout_label,
            window_size=window_size,
        )

    def parse_prediction(self, raw: str) -> str:
        """Extract 'yes' or 'no' from raw model output.

        Strategy: check the last non-empty line first (models are instructed
        to put the answer there), then fall back to scanning the full text.
        Returns 'unknown' if neither 'yes' nor 'no' is found.
        """
        text = raw.strip().lower()
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        for line in reversed(lines):
            if re.search(r"\byes\b", line):
                return "yes"
            if re.search(r"\bno\b", line):
                return "no"

        # Fallback: anywhere in the full response
        if re.search(r"\byes\b", text):
            return "yes"
        if re.search(r"\bno\b", text):
            return "no"

        return "unknown"

    def aggregate_metrics(
        self,
        ground_truths: list[str],
        predictions: list[str],
    ) -> dict[str, float | int]:
        """Compute binary classification metrics, excluding 'unknown' predictions.

        Returns all binary_metrics keys plus 'n_unknown' and 'parse_rate'.
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
        metrics = binary_metrics(list(gts), list(preds))
        metrics["n_unknown"] = n_unknown
        metrics["parse_rate"] = round(len(filtered) / len(ground_truths), 4)
        return metrics
