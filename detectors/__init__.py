#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class DetectionResult:
    event_type: Literal["trend", "spike", "spike_multi"]
    start_minute: Optional[int] = None
    end_minute: Optional[int] = None
    direction: Optional[Literal["increasing", "decreasing"]] = None
    spike_minutes: Optional[list[int]] = None
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.event_type == "trend":
            if self.start_minute is None or self.end_minute is None or self.direction is None:
                raise ValueError("trend requires start_minute, end_minute, and direction")
        elif not self.spike_minutes:
            raise ValueError(f"{self.event_type} requires spike_minutes")

    def template_vars(self) -> dict:
        if self.event_type == "trend":
            return {"direction": self.direction, "start": self.start_minute, "end": self.end_minute}
        return {"times": _format_times(self.spike_minutes)}

    @property
    def window(self) -> tuple[int, int]:
        if self.event_type == "trend":
            return (self.start_minute, self.end_minute)
        return (min(self.spike_minutes), max(self.spike_minutes))


def _format_times(minutes: list[int]) -> str:
    if len(minutes) == 1:
        return str(minutes[0])
    if len(minutes) == 2:
        return f"{minutes[0]} and {minutes[1]}"
    return ", ".join(str(m) for m in minutes[:-1]) + f", and {minutes[-1]}"


class StructuralDetector(abc.ABC):
    def __init__(self, filter_zeros: bool = False):
        self.filter_zeros = filter_zeros

    def detect(self, raw_signal: np.ndarray) -> list[DetectionResult]:
        mask = ~np.isnan(raw_signal)
        if self.filter_zeros:
            mask &= raw_signal != 0
        indices = np.where(mask)[0].astype(np.int32)
        filtered_signal = raw_signal[indices].astype(float)
        if len(filtered_signal) == 0:
            return []
        return self._detect(filtered_signal, indices)

    @abc.abstractmethod
    def _detect(self, filtered_signal: np.ndarray, indices: np.ndarray) -> list[DetectionResult]: ...
