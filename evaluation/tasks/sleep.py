#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Sleep evaluation task — stub, pending PI alignment.

Ground truth will come from sleep:asleep binary channel (Apple Watch).
Design questions to resolve:
  - Window granularity: 30-min epochs (standard in sleep science) or 60-min?
  - Binary (asleep/awake) or multi-class (asleep/inbed/awake)?
  - Minimum night coverage required to include a recording?
"""
from __future__ import annotations

from evaluation.tasks.base import EvalSample, EvalTask


class SleepTask(EvalTask):
    """Placeholder — not yet implemented."""

    name = "sleep"

    def build_prompt(self, sample: EvalSample) -> str:
        raise NotImplementedError("SleepTask is not yet implemented.")

    def parse_prediction(self, raw: str) -> str:
        raise NotImplementedError("SleepTask is not yet implemented.")

    def aggregate_metrics(
        self,
        ground_truths: list[str],
        predictions: list[str],
    ) -> dict[str, float | int]:
        raise NotImplementedError("SleepTask is not yet implemented.")
