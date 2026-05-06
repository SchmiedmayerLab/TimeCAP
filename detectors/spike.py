#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from detectors import DetectionResult, StructuralDetector


class SpikeDetector(StructuralDetector):
    """Detects peaks using a minimal SciPy-backed configuration."""

    def __init__(
        self,
        filter_zeros: bool = False,
        min_height: float | None = None,
        min_prominence: float = 0.0,
        min_distance: int = 1,
        min_width: int = 1,
        top_k: int | None = None,
        max_cluster_peaks: int = 2,
        soft_top_k_margin: float = 0.90,
    ) -> None:
        super().__init__(filter_zeros=filter_zeros)
        self.min_height = min_height
        self.min_prominence = min_prominence
        self.min_distance = max(1, min_distance)
        self.min_width = max(1, min_width)
        self.top_k = top_k
        self.max_cluster_peaks = max(1, max_cluster_peaks)
        self.soft_top_k_margin = min(max(float(soft_top_k_margin), 0.0), 1.0)

    def _detect(self, filtered_signal: np.ndarray, indices: np.ndarray) -> list[DetectionResult]:
        peaks, properties = find_peaks(
            filtered_signal,
            height=self.min_height,
            prominence=self.min_prominence,
            distance=self.min_distance,
            width=self.min_width,
        )
        if len(peaks) == 0:
            return []

        prominences = properties["prominences"].astype(float)
        widths = properties.get("widths", np.ones_like(peaks, dtype=float)).astype(float)
        scores = prominences + 0.25 * widths
        raw_minutes = indices[peaks]
        order = np.lexsort((raw_minutes, -scores))
        if self.top_k is not None:
            order = order[:self.top_k]

        minutes = [int(m) for m in raw_minutes[order]]
        max_score = float(scores[order[0]])
        event_type = "spike" if len(minutes) == 1 else "spike_multi"
        return [DetectionResult(event_type=event_type, spike_minutes=minutes, score=max_score)]
