#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Core abstractions for evaluation tasks.

EvalSample  — a single labelled sample (windowed Recording + ground truth).
EvalTask    — abstract base for task-specific prompt building and metric aggregation.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

from timef.schema import Recording


@dataclass(slots=True)
class EvalSample:
    """A single evaluation sample.

    Attributes:
        sample_id:    Unique identifier (used in result files for traceability).
        recording:    Windowed Recording fed to the model as visual signal plots.
                      Workout/label channels are intentionally excluded.
        ground_truth: Expected answer string, e.g. "yes" / "no".
        metadata:     Task-specific extras (workout_type, window_start, date, …).
    """
    sample_id: str
    recording: Recording
    ground_truth: str
    metadata: dict = field(default_factory=dict)


class EvalTask(abc.ABC):
    """Abstract base class for an evaluation task.

    A task owns three responsibilities:
        1. build_prompt    — turn an EvalSample into a text prompt for the model.
        2. parse_prediction — extract a canonical answer from raw model output.
        3. aggregate_metrics — compute dataset-level metrics from all predictions.

    Subclasses must set the ``name`` class attribute and implement all three methods.
    """

    name: str  # set by each concrete subclass, used for result file paths

    @abc.abstractmethod
    def build_prompt(self, sample: EvalSample) -> str:
        """Return the text prompt to send to the model for this sample."""
        ...

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
