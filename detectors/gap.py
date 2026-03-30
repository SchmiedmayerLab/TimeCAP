#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np

from detectors import DetectionResult, StructuralDetector


class GapDetector(StructuralDetector):
    """Detects contiguous runs of zero or NaN in a signal."""

    def __init__(self, min_duration: int = 30) -> None:
        super().__init__(filter_zeros=False)
        self.min_duration = min_duration

    def detect(self, series: np.ndarray) -> list[DetectionResult]:
        gap_mask = np.isnan(series) | (series == 0)
        if not gap_mask.any():
            return []
        padded = np.concatenate(([False], gap_mask, [False]))
        diffs = np.diff(padded.astype(np.int8))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        lengths = ends - starts
        mask = lengths >= self.min_duration
        return [
            DetectionResult(event_type="gap", start_minute=int(s), end_minute=int(e - 1))
            for s, e in zip(starts[mask], ends[mask])
        ]

    def _detect(self, series: np.ndarray, indices: np.ndarray) -> list[DetectionResult]:
        return []
