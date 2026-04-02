#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from detectors import DetectionResult, StructuralDetector


class SpikeDetector(StructuralDetector):
    """Detects spikes and drops.
    """

    def __init__(
        self,
        filter_zeros: bool = False,
        prominence_scale: float = 3.0,
        min_prominence: float = 0.0,
        min_distance: int = 1,
        min_width: int = 1,
        top_k: int | None = None,
        smooth_window: int = 1,
    ) -> None:
        super().__init__(filter_zeros=filter_zeros)
        self.prominence_scale = prominence_scale
        self.min_prominence = min_prominence
        self.min_distance = max(1, min_distance)
        self.min_width = max(1, min_width)
        self.top_k = top_k
        self.smooth_window = max(1, smooth_window)

    def _detect(self, filtered_signal: np.ndarray, indices: np.ndarray) -> list[DetectionResult]:
        smoothed_signal = self._smooth(filtered_signal)
        prominence = self._prominence_threshold(filtered_signal)  # threshold on raw/filtered, not smoothed
        if prominence <= 0:
            return []

        results: list[DetectionResult] = []
        seen_minutes: set[int] = set()

        for peak_idx, peak_score in self._select_peaks(smoothed_signal, prominence):
            minute = int(indices[peak_idx])
            if minute in seen_minutes:
                continue
            results.append(DetectionResult(event_type="spike", spike_minute=minute, score=peak_score))
            seen_minutes.add(minute)

        for peak_idx, peak_score in self._select_peaks(-smoothed_signal, prominence):
            minute = int(indices[peak_idx])
            if minute in seen_minutes:
                continue
            results.append(DetectionResult(event_type="drop", spike_minute=minute, score=peak_score))
            seen_minutes.add(minute)

        results.sort(key=lambda result: int(result.spike_minute))
        return results

    def _smooth(self, filtered_signal: np.ndarray) -> np.ndarray:
        if self.smooth_window <= 1:
            return filtered_signal

        kernel = np.ones(self.smooth_window, dtype=float) / float(self.smooth_window)
        return np.convolve(filtered_signal, kernel, mode="same")

    def _select_peaks(self, smoothed_signal: np.ndarray, prominence: float) -> list[tuple[int, float]]:
        peaks, properties = find_peaks(
            smoothed_signal,
            prominence=prominence,
            distance=self.min_distance,
            width=self.min_width,
        )
        if len(peaks) == 0:
            return []

        prominences = properties["prominences"].astype(float)
        order = np.argsort(prominences)[::-1]
        if self.top_k is not None:
            order = order[: self.top_k]

        return [(int(peaks[idx]), float(prominences[idx])) for idx in order]

    def _prominence_threshold(self, filtered_signal: np.ndarray) -> float:
        centered = filtered_signal - np.median(filtered_signal)
        mad = float(np.median(np.abs(centered)))
        if mad > 0:
            scale = 1.4826 * mad
            return max(self.min_prominence, self.prominence_scale * scale)

        spread = float(np.percentile(filtered_signal, 95) - np.percentile(filtered_signal, 5))
        if spread <= 1e-12:
            return 0.0
        return max(self.min_prominence, 0.5 * spread)
