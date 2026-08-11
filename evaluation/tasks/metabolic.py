#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Metabolic prediction evaluation task.

The dataset owns prompt wording and answer options. This task is responsible
only for parsing the model output and dispatching to the correct metric family.
Binary labels use binary classification metrics, ordinal labels use ordinal
metrics, and categorical labels use multiclass metrics.
"""
from __future__ import annotations

import re  # used in parse_prediction

from evaluation.metrics.classification import (
    binary_metrics,
    multiclass_metrics,
    ordinal_metrics,
)
from evaluation.tasks.base import EvalTask
from time_series_datasets.mhc_label_lookup import (
    LABEL_TYPES,
    get_metabolic_label_config,
)
from time_series_datasets.mhc_metabolic_qa_dataset import MHCMetabolicQADataset


class MetabolicTask(EvalTask):
    """Parse predictions and aggregate metrics for one metabolic label."""

    name = "metabolic"
    dataset_cls = MHCMetabolicQADataset

    def __init__(self, label_column: str) -> None:
        if label_column not in LABEL_TYPES:
            raise ValueError(
                f"Unknown label_column '{label_column}'. "
                f"All MHC labels (from mhc_label_lookup.LABEL_TYPES): {sorted(LABEL_TYPES)}"
            )
        cfg = get_metabolic_label_config(label_column)
        self._label_column = label_column
        self._label_type = LABEL_TYPES[label_column]
        self._values = list(cfg["values"])
        self._match_values = sorted(self._values, key=len, reverse=True)
        self._positive_label = self._values[0]
        self.name = label_column

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

        # Prefer the explicit "Answer: <value>" suffix when present.
        answer_match = re.search(
            r"Answer:\s*([^\n.,;!?]+)", text, re.IGNORECASE
        )
        if answer_match:
            candidate = answer_match.group(1).strip().rstrip(".,!?;:")
            for value in self._match_values:
                if candidate.lower() == value.lower():
                    return value
            return "unknown"

        def _match(s: str) -> str | None:
            if self._REFUSAL_PATTERNS.search(s):
                return None
            lower = s.lower()
            for value in self._match_values:
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
            for value in self._match_values:
                if re.search(r"\b" + re.escape(value.lower()) + r"\b", lower):
                    return value

        return "unknown"

    def aggregate_metrics(
        self,
        ground_truths: list[str],
        predictions: list[str],
    ) -> dict[str, float | int]:
        """Compute label-type-aware metrics, excluding unknown predictions."""
        if len(ground_truths) != len(predictions):
            raise ValueError(
                "Length mismatch: "
                f"{len(ground_truths)} ground truths vs {len(predictions)} predictions"
            )

        n_unknown = sum(1 for p in predictions if p == "unknown")
        filtered = [
            (gt, pred)
            for gt, pred in zip(ground_truths, predictions, strict=True)
            if pred != "unknown"
        ]

        if not filtered:
            return self._empty_metrics(ground_truths, n_unknown)

        gts, preds = zip(*filtered)
        if self._label_type == "binary":
            metrics = binary_metrics(
                list(gts), list(preds), positive_label=self._positive_label
            )
        elif self._label_type == "ordinal":
            metrics = ordinal_metrics(list(gts), list(preds), labels=self._values)
        else:
            metrics = multiclass_metrics(list(gts), list(preds), labels=self._values)
        metrics["n_unknown"] = n_unknown
        metrics["parse_rate"] = round(len(filtered) / len(ground_truths), 4)
        return metrics

    def _empty_metrics(
        self,
        ground_truths: list[str],
        n_unknown: int,
    ) -> dict[str, float | int]:
        if self._label_type == "binary":
            n_positive_gt = sum(1 for gt in ground_truths if gt == self._positive_label)
            return {
                "f1": 0.0, "f1_se": 0.0,
                "precision": 0.0, "precision_se": 0.0,
                "recall": 0.0, "recall_se": 0.0,
                "accuracy": 0.0, "accuracy_se": 0.0,
                "n_samples": len(ground_truths),
                "n_positive_gt": n_positive_gt,
                "n_positive_pred": 0,
                "n_unknown": n_unknown,
                "parse_rate": 0.0,
            }
        if self._label_type == "ordinal":
            return {
                "spearman_r": 0.0,
                "spearman_r_se": 0.0,
                "qwk": 0.0,
                "qwk_se": 0.0,
                "mae_ordinal": 0.0,
                "mae_ordinal_se": 0.0,
                "n_samples": len(ground_truths),
                "n_classes": len(self._values),
                "n_unknown": n_unknown,
                "parse_rate": 0.0,
            }
        return {
            "macro_f1": 0.0,
            "macro_f1_se": 0.0,
            "accuracy": 0.0,
            "accuracy_se": 0.0,
            "n_samples": len(ground_truths),
            "n_classes": len(self._values),
            "n_unknown": n_unknown,
            "parse_rate": 0.0,
        }
