from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class DetectionResult:
    event_type: str                    # "trend" | "spike" | "drop"
    start_minute: Optional[int] = None
    end_minute: Optional[int] = None
    direction: Optional[str] = None    # "increasing" | "decreasing"
    spike_minute: Optional[int] = None

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

    def detect(self, series: np.ndarray) -> List[DetectionResult]:
        mask = ~np.isnan(series)
        if self.filter_zeros:
            mask &= series > 0
        indices = np.where(mask)[0].astype(np.int32)
        series = series[indices].astype(float)
        if len(series) == 0:
            return []
        return self._detect(series, indices)

    @abc.abstractmethod
    def _detect(self, series: np.ndarray, indices: np.ndarray) -> List[DetectionResult]: ...
