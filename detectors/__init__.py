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
    event_type: Literal["trend", "spike", "drop"]
    start_minute: Optional[int] = None
    end_minute: Optional[int] = None
    direction: Optional[Literal["increasing", "decreasing"]] = None
    spike_minute: Optional[int] = None

    def __post_init__(self) -> None:
        if self.event_type == "trend":
            if self.start_minute is None or self.end_minute is None or self.direction is None:
                raise ValueError("trend requires start_minute, end_minute, and direction")
        elif self.spike_minute is None:
            raise ValueError(f"{self.event_type} requires spike_minute")

    def template_vars(self) -> dict:
        if self.event_type == "trend":
            return {"direction": self.direction, "start": self.start_minute, "end": self.end_minute}
        return {"minute": self.spike_minute}

    @property
    def window(self) -> tuple[int, int]:
        if self.event_type == "trend":
            return (self.start_minute, self.end_minute)
        return (self.spike_minute, self.spike_minute)


class StructuralDetector(abc.ABC):
    def __init__(self, filter_zeros: bool = False):
        self.filter_zeros = filter_zeros

    def detect(self, series: np.ndarray) -> list[DetectionResult]:
        mask = ~np.isnan(series)
        if self.filter_zeros:
            mask &= series != 0
        indices = np.where(mask)[0].astype(np.int32)
        series = series[indices].astype(float)
        if len(series) == 0:
            return []
        return self._detect(series, indices)

    @abc.abstractmethod
    def _detect(self, series: np.ndarray, indices: np.ndarray) -> list[DetectionResult]: ...
