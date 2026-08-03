#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""MIMIC-IV in-hospital mortality classification task."""
from __future__ import annotations

import re

from evaluation.tasks.base import EvalTask

MIMIC_MORTALITY_VALUES = ["survived", "died"]
MIMIC_TRUEFALSE_VALUES = ["True", "False"]


class MIMICMortalityTask(EvalTask):
    """Parse and score MIMIC-IV mortality predictions.

    ``label_space`` selects the vocabulary the model is asked to answer in.
    ``native`` uses survived/died; ``truefalse`` asks the same question with
    True/False, which is a space an MHC-trained checkpoint has actually seen.
    The task must be built with the same value the dataset was, or every
    prediction parses as ``unknown`` — ``run_mimic_eval.py`` passes one flag to
    both so they cannot diverge.
    """

    positive_label = "died"

    def __init__(self, label_space: str = "native") -> None:
        if label_space == "truefalse":
            self.values = list(MIMIC_TRUEFALSE_VALUES)
            self.positive_label = "True"
            self.name = "mimic_mortality_truefalse"
        elif label_space == "native":
            self.values = list(MIMIC_MORTALITY_VALUES)
            self.positive_label = "died"
            self.name = "mimic_mortality"
        else:
            raise ValueError(
                f"label_space must be 'native' or 'truefalse', got {label_space!r}"
            )
        self.label_space = label_space
        self._match_values = sorted(self.values, key=len, reverse=True)
    _REFUSAL_PATTERNS = re.compile(
        r"\b(i('m| am) sorry|i can'?t|i cannot|unable to|not possible|"
        r"i('m| am) not able|it('s| is) not possible)\b",
        re.IGNORECASE,
    )

    def parse_prediction(self, raw: str) -> str:
        text = raw.strip()
        answer_match = re.search(
            r"Answer:\s*([^\n.,;!?]+)",
            text,
            re.IGNORECASE,
        )
        if answer_match:
            candidate = answer_match.group(1).strip().rstrip(".,!?;:").lower()
            for value in self._match_values:
                if candidate == value.lower():
                    return value
            return "unknown"

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            hit = self._match_label(line)
            if hit is not None:
                return hit

        return self._match_label(text) or "unknown"

    def _match_label(self, text: str) -> str | None:
        if self._REFUSAL_PATTERNS.search(text):
            return None
        lower = text.lower()
        for value in self._match_values:
            if re.search(r"\b" + re.escape(value.lower()) + r"\b", lower):
                return value
        return None

    def aggregate_metrics(
        self,
        ground_truths: list[str],
        predictions: list[str],
    ) -> dict[str, float | int | dict]:
        if len(ground_truths) != len(predictions):
            raise ValueError(
                "Length mismatch: "
                f"{len(ground_truths)} ground truths vs {len(predictions)} predictions"
            )

        n_unknown = sum(1 for pred in predictions if pred == "unknown")
        filtered = [
            (gt, pred)
            for gt, pred in zip(ground_truths, predictions)
            if pred != "unknown"
        ]
        if not filtered:
            return self._empty_metrics(ground_truths, n_unknown)

        gts, preds = zip(*filtered)
        return _binary_report(
            list(gts),
            list(preds),
            labels=self.values,
            positive_label=self.positive_label,
            n_total=len(ground_truths),
            n_unknown=n_unknown,
        )

    def _empty_metrics(
        self,
        ground_truths: list[str],
        n_unknown: int,
    ) -> dict[str, float | int | dict]:
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "n_samples": len(ground_truths),
            "n_evaluated": 0,
            "n_positive_gt": sum(1 for gt in ground_truths if gt == self.positive_label),
            "n_positive_pred": 0,
            "n_unknown": n_unknown,
            "parse_rate": 0.0,
            "confusion_matrix": _empty_confusion_matrix(self.values),
        }


def _binary_report(
    ground_truths: list[str],
    predictions: list[str],
    labels: list[str],
    positive_label: str,
    n_total: int,
    n_unknown: int,
) -> dict[str, float | int | dict]:
    tp = sum(
        1 for gt, pred in zip(ground_truths, predictions)
        if gt == positive_label and pred == positive_label
    )
    fp = sum(
        1 for gt, pred in zip(ground_truths, predictions)
        if gt != positive_label and pred == positive_label
    )
    fn = sum(
        1 for gt, pred in zip(ground_truths, predictions)
        if gt == positive_label and pred != positive_label
    )
    n_correct = sum(
        1 for gt, pred in zip(ground_truths, predictions)
        if gt == pred
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "accuracy": round(float(n_correct / len(ground_truths)), 6),
        "f1": round(float(f1), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "n_samples": int(n_total),
        "n_evaluated": int(len(ground_truths)),
        "n_positive_gt": sum(1 for gt in ground_truths if gt == positive_label),
        "n_positive_pred": sum(1 for pred in predictions if pred == positive_label),
        "n_unknown": int(n_unknown),
        "parse_rate": round(float(len(ground_truths) / n_total), 6),
        "confusion_matrix": _confusion_matrix(ground_truths, predictions, labels),
    }


def _confusion_matrix(
    ground_truths: list[str],
    predictions: list[str],
    labels: list[str],
) -> dict[str, dict[str, int]]:
    matrix = _empty_confusion_matrix(labels)
    for gt, pred in zip(ground_truths, predictions):
        if gt in matrix and pred in matrix[gt]:
            matrix[gt][pred] += 1
    return matrix


def _empty_confusion_matrix(labels: list[str]) -> dict[str, dict[str, int]]:
    return {gt: {pred: 0 for pred in labels} for gt in labels}
