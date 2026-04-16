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
Supported metric families are binary, multiclass, and ordinal classification.

Example
-------
>>> from evaluation.metrics.classification import (
...     binary_metrics, multiclass_metrics, ordinal_metrics
... )
>>> binary = binary_metrics(["yes", "no", "yes"], ["yes", "yes", "yes"])
>>> multiclass = multiclass_metrics(["a", "b", "c"], ["a", "b", "b"])
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
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


def multiclass_metrics(
    ground_truths: list[str],
    predictions: list[str],
    labels: list[str] | None = None,
) -> dict[str, float | int]:
    """Accuracy and macro-F1 with bootstrap SEs for categorical labels."""
    y_true = np.asarray(ground_truths, dtype=object)
    y_pred = np.asarray(predictions, dtype=object)
    label_set = labels if labels is not None else sorted({*ground_truths, *predictions})

    def _macro_f1(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(
            f1_score(yt, yp, labels=label_set, average="macro", zero_division=0)
        )

    def _accuracy(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(accuracy_score(yt, yp))

    return {
        "macro_f1":      _macro_f1(y_true, y_pred),
        "macro_f1_se":   bootstrap_se(y_true, y_pred, _macro_f1),
        "accuracy":      _accuracy(y_true, y_pred),
        "accuracy_se":   bootstrap_se(y_true, y_pred, _accuracy),
        "n_samples":     int(len(ground_truths)),
        "n_classes":     int(len(label_set)),
    }


def ordinal_metrics(
    ground_truths: list[str],
    predictions: list[str],
    labels: list[str],
) -> dict[str, float | int]:
    """Spearman, quadratic weighted kappa, and MAE for ordered labels."""
    label_to_rank = {label: idx for idx, label in enumerate(labels)}
    y_true = np.asarray([label_to_rank[g] for g in ground_truths], dtype=np.int64)
    y_pred = np.asarray([label_to_rank[p] for p in predictions], dtype=np.int64)

    def _spearman(yt: np.ndarray, yp: np.ndarray) -> float:
        if len(np.unique(yt)) < 2 or len(np.unique(yp)) < 2:
            return float("nan")
        score = spearmanr(yt, yp)[0]
        return float(score) if score is not None else float("nan")

    def _qwk(yt: np.ndarray, yp: np.ndarray) -> float:
        if len(np.unique(yt)) < 2 and len(np.unique(yp)) < 2:
            return float("nan")
        return float(
            cohen_kappa_score(
                yt,
                yp,
                labels=list(range(len(labels))),
                weights="quadratic",
            )
        )

    def _mae(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(mean_absolute_error(yt, yp))

    spearman_val = _spearman(y_true, y_pred)
    qwk_val = _qwk(y_true, y_pred)
    mae_val = _mae(y_true, y_pred)

    return {
        "spearman_r": float(spearman_val),
        "spearman_r_se": bootstrap_se(y_true, y_pred, _spearman),
        "qwk": float(qwk_val),
        "qwk_se": bootstrap_se(y_true, y_pred, _qwk),
        "mae_ordinal": float(mae_val),
        "mae_ordinal_se": bootstrap_se(y_true, y_pred, _mae),
        "n_samples": int(len(ground_truths)),
        "n_classes": int(len(labels)),
    }
