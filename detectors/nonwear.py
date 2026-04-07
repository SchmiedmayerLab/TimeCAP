#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np

from detectors import DetectionResult, StructuralDetector


class NonWearDetector(StructuralDetector):
    """Detect long missing-data runs that likely indicate the wearable was not worn."""

    def __init__(self, min_length: int = 30) -> None:
        super().__init__(filter_zeros=False)
        self.min_length = max(1, int(min_length))

    def detect(self, raw_signal: np.ndarray) -> list[DetectionResult]:
        nonwear_mask = np.isnan(np.asarray(raw_signal, dtype=float))
        if not nonwear_mask.any():
            return []
        return self._detect(nonwear_mask.astype(float), np.arange(len(nonwear_mask), dtype=np.int32))

    def _detect(self, filtered_signal: np.ndarray, indices: np.ndarray) -> list[DetectionResult]:
        nonwear_mask = filtered_signal > 0
        padded = np.concatenate(([False], nonwear_mask, [False]))
        diffs = np.diff(padded.astype(np.int8))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        results: list[DetectionResult] = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            duration = end - start
            if duration < self.min_length:
                continue
            results.append(
                DetectionResult(
                    event_type="nonwear",
                    start_minute=int(indices[start]),
                    end_minute=int(indices[end - 1]),
                    score=float(duration),
                )
            )
        return results
