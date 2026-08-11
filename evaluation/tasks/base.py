#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Core abstraction for evaluation tasks."""
from __future__ import annotations

import abc


class EvalTask(abc.ABC):
    """Abstract base class for an evaluation task.

    A task owns two responsibilities:
        1. parse_prediction  — extract a canonical answer from raw model output.
        2. aggregate_metrics — compute dataset-level metrics from all predictions.

    Subclasses must set the ``name`` class attribute and implement both methods.
    """

    name: str  # set by each concrete subclass, used for result file paths

    @abc.abstractmethod
    def parse_prediction(self, raw: str) -> str:
        """Extract a canonical answer token from raw model output."""
        ...

    @abc.abstractmethod
    def aggregate_metrics(
        self,
        ground_truths: list[str],
        predictions: list[str],
    ) -> dict[str, float]:
        """Compute dataset-level metrics from ground truths and parsed predictions."""
        ...
