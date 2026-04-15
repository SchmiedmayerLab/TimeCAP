#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Classification metrics with bootstrap standard errors.

All functions are pure (no state) and return a flat dict of floats so results
can be serialised directly to JSON. Bootstrap SEs use 1,000 resamples and
report the standard deviation of bootstrapped metric values as the SE estimate.

Example
-------
>>> from evaluation.metrics.classification import binary_metrics
>>> metrics = binary_metrics(["yes", "no", "yes"], ["yes", "yes", "yes"])
>>> metrics["f1"], metrics["f1_se"]
(0.6666..., ...)
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

_N_BOOTSTRAP = 1_000
_BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------

def bootstrap_se(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = _N_BOOTSTRAP,
    seed: int = _BOOTSTRAP_SEED,
) -> float:
    """Compute bootstrap standard error for a metric.

    Args:
        y_true:    Ground-truth label array.
        y_pred:    Predicted label array.
        metric_fn: f(y_true, y_pred) -> float.
        n_boot:    Number of bootstrap resamples.
        seed:      Random seed for reproducibility.

    Returns:
        Standard error (std of bootstrapped metric values).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])
    return float(np.nanstd(scores, ddof=1))


# ---------------------------------------------------------------------------
# Binary classification metrics
# ---------------------------------------------------------------------------

def binary_metrics(
    ground_truths: list[str],
    predictions: list[str],
    positive_label: str = "yes",
) -> dict[str, float | int]:
    """F1, Precision, Recall, Accuracy with bootstrap SEs for binary labels.

    Args:
        ground_truths:  List of canonical ground-truth strings.
        predictions:    List of parsed model predictions (same length).
        positive_label: The string that counts as the positive class.

    Returns:
        Flat dict: f1, f1_se, precision, precision_se, recall, recall_se,
        accuracy, accuracy_se, n_samples, n_positive_gt, n_positive_pred.
    """
    y_true = np.array([1 if g == positive_label else 0 for g in ground_truths])
    y_pred = np.array([1 if p == positive_label else 0 for p in predictions])

    def _f1(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(f1_score(yt, yp, zero_division=0))

    def _precision(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(precision_score(yt, yp, zero_division=0))

    def _recall(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(recall_score(yt, yp, zero_division=0))

    def _accuracy(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(accuracy_score(yt, yp))

    return {
        "f1":            _f1(y_true, y_pred),
        "f1_se":         bootstrap_se(y_true, y_pred, _f1),
        "precision":     _precision(y_true, y_pred),
        "precision_se":  bootstrap_se(y_true, y_pred, _precision),
        "recall":        _recall(y_true, y_pred),
        "recall_se":     bootstrap_se(y_true, y_pred, _recall),
        "accuracy":      _accuracy(y_true, y_pred),
        "accuracy_se":   bootstrap_se(y_true, y_pred, _accuracy),
        "n_samples":     int(len(ground_truths)),
        "n_positive_gt": int(y_true.sum()),
        "n_positive_pred": int(y_pred.sum()),
    }
