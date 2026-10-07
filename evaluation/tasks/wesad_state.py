#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""WESAD state classification tasks: 4-way, and binary stress detection."""
from __future__ import annotations

import re

from evaluation.metrics.classification import binary_metrics
from evaluation.tasks.base import EvalTask

WESAD_STATE_VALUES = ["baseline", "stress", "amusement", "meditation"]
WESAD_STRESS_VALUES = ["True", "False"]


class WESADStateTask(EvalTask):
    """Parse and score WESAD state predictions."""

    name = "wesad_state"
    values = WESAD_STATE_VALUES
    _match_values = sorted(values, key=len, reverse=True)
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
            for gt, pred in zip(ground_truths, predictions, strict=True)
            if pred != "unknown"
        ]
        if not filtered:
            return self._empty_metrics(ground_truths, n_unknown)

        gts, preds = zip(*filtered)
        return _classification_report(
            list(gts),
            list(preds),
            labels=self.values,
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
            "macro_f1": 0.0,
            "n_samples": len(ground_truths),
            "n_evaluated": 0,
            "n_unknown": n_unknown,
            "parse_rate": 0.0,
            "per_class": {
                label: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
                for label in self.values
            },
            "confusion_matrix": _empty_confusion_matrix(self.values),
        }


class WESADBinaryTask(WESADStateTask):
    """One-vs-rest: is this window the given state, True or False?

    The 4-way task is unscorable with an MHC-trained checkpoint. Its label space
    was never in training, and such a model answers ``True``/``False`` whatever
    options the prompt offers — the identical collapse MHC's own unseen
    ``blood_pressure_categories`` shows (parse rate 0.0000 there, on the
    canonical prompt builder). Reducing to stress-vs-rest puts the label space
    back in distribution so the output can be parsed at all.

    This is a recognised WESAD framing rather than a workaround: the dataset's
    own protocol contrasts a stress condition against non-stress ones, and
    binary stress detection is reported alongside the 3-way split in the WESAD
    literature. The other three states collapse into the negative class.

    Running one of these per state and combining them (see
    ``scripts/combine_wesad_ovr.py``) recovers a 4-way prediction without ever
    asking the model to emit a label it has not seen.
    """

    values = WESAD_STRESS_VALUES
    _match_values = sorted(WESAD_STRESS_VALUES, key=len, reverse=True)
    _positive_label = "True"

    def __init__(self, positive_state: str = "stress") -> None:
        if positive_state not in WESAD_STATE_VALUES:
            raise ValueError(
                f"positive_state must be one of {WESAD_STATE_VALUES}, "
                f"got {positive_state!r}"
            )
        self.positive_state = positive_state
        # Results land in results/wesad_<state>/, keeping each one-vs-rest run
        # in its own tree so the combiner can find them by name.
        self.name = f"wesad_{positive_state}"

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
            for gt, pred in zip(ground_truths, predictions, strict=True)
            if pred != "unknown"
        ]
        if filtered:
            gts, preds = zip(*filtered)
            metrics: dict[str, float | int | dict] = binary_metrics(
                list(gts), list(preds), positive_label=self._positive_label
            )
        else:
            metrics = {
                "f1": 0.0, "f1_se": 0.0,
                "precision": 0.0, "precision_se": 0.0,
                "recall": 0.0, "recall_se": 0.0,
                "accuracy": 0.0, "accuracy_se": 0.0,
                "n_samples": 0, "n_positive_gt": 0, "n_positive_pred": 0,
            }

        # Same split as MetabolicTask: n_samples counts scored predictions,
        # n_evaluated counts what the model was run on — the denominator a
        # results table needs.
        metrics["n_unknown"] = n_unknown
        metrics["parse_rate"] = (
            round(len(filtered) / len(ground_truths), 6) if ground_truths else 0.0
        )
        metrics["n_evaluated"] = len(ground_truths)
        return metrics


def _classification_report(
    ground_truths: list[str],
    predictions: list[str],
    labels: list[str],
    n_total: int,
    n_unknown: int,
) -> dict[str, float | int | dict]:
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    n_correct = sum(
        1 for gt, pred in zip(ground_truths, predictions, strict=True)
        if gt == pred
    )
    for label in labels:
        tp = sum(
            1 for gt, pred in zip(ground_truths, predictions, strict=True)
            if gt == label and pred == label
        )
        fp = sum(
            1 for gt, pred in zip(ground_truths, predictions, strict=True)
            if gt != label and pred == label
        )
        fn = sum(
            1 for gt, pred in zip(ground_truths, predictions, strict=True)
            if gt == label and pred != label
        )
        support = sum(1 for gt in ground_truths if gt == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[label] = {
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "support": int(support),
        }

    return {
        "accuracy": round(float(n_correct / len(ground_truths)), 6),
        "macro_f1": round(float(sum(f1_values) / len(f1_values)), 6),
        "n_samples": int(n_total),
        "n_evaluated": int(len(ground_truths)),
        "n_unknown": int(n_unknown),
        "parse_rate": round(float(len(ground_truths) / n_total), 6),
        "per_class": per_class,
        "confusion_matrix": _confusion_matrix(ground_truths, predictions, labels),
    }


def _confusion_matrix(
    ground_truths: list[str],
    predictions: list[str],
    labels: list[str],
) -> dict[str, dict[str, int]]:
    matrix = _empty_confusion_matrix(labels)
    for gt, pred in zip(ground_truths, predictions, strict=True):
        if gt in matrix and pred in matrix[gt]:
            matrix[gt][pred] += 1
    return matrix


def _empty_confusion_matrix(labels: list[str]) -> dict[str, dict[str, int]]:
    return {gt: {pred: 0 for pred in labels} for gt in labels}
