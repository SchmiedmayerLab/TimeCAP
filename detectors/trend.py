#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from detectors import DetectionResult, StructuralDetector


@dataclass
class _TrendSegment:
    direction: str
    start_minute: int
    end_minute: int
    score: float


class TrendDetector(StructuralDetector):
    """Detects increasing/decreasing trends across multiple relative window sizes."""

    def __init__(
        self,
        filter_zeros: bool = False,
        window_sizes: tuple[int, ...] | None = None,
        window_fracs: tuple[float, ...] = (0.1, 0.25, 1.0),
        min_window: int = 12,
        max_window: int | None = None,
        stride_frac: float = 0.25,
        min_coverage: float = 0.75,
        min_effect: float = 1.25,
        min_r2: float = 0.5,
        min_span: int = 10,
        merge_gap: int = 5,
    ) -> None:
        super().__init__(filter_zeros=filter_zeros)
        self.window_sizes = window_sizes
        self.window_fracs = tuple(window_fracs)
        self.min_window = max(3, min_window)
        self.max_window = max_window
        self.stride_frac = stride_frac
        self.min_coverage = min_coverage
        self.min_effect = min_effect
        self.min_r2 = min_r2
        self.min_span = min_span
        self.merge_gap = merge_gap

    def _detect(self, filtered_signal: np.ndarray, indices: np.ndarray) -> list[DetectionResult]:
        window_sizes = self._resolve_window_sizes(len(filtered_signal))
        segments: list[_TrendSegment] = []
        for window_size in window_sizes:
            stride = max(1, int(round(window_size * self.stride_frac)))
            for start_idx in self._window_starts(len(filtered_signal), window_size, stride):
                end_idx = start_idx + window_size
                segment = self._classify_window(filtered_signal[start_idx:end_idx], indices[start_idx:end_idx])
                if segment is not None:
                    segments.append(segment)

        merged = self._resolve_containment(self._merge_segments(segments))
        resolved = self._split_opposite_overlaps(merged, filtered_signal, indices)
        # score reflects the pre-split fit; segments trimmed by _split_opposite_overlaps are not refit.
        return [
            DetectionResult(
                event_type="trend",
                start_minute=segment.start_minute,
                end_minute=segment.end_minute,
                direction=segment.direction,
                score=segment.score,
            )
            for segment in resolved
            if segment.end_minute - segment.start_minute >= self.min_span
        ]

    def _resolve_window_sizes(self, n_samples: int) -> list[int]:
        sizes = set(self.window_sizes or ())
        for frac in self.window_fracs:
            if frac <= 0:
                continue
            sizes.add(int(round(n_samples * frac)))

        resolved = []
        max_window = self.max_window or n_samples
        for size in sorted(sizes):
            size = max(self.min_window, min(size, max_window, n_samples))
            if size <= n_samples:
                resolved.append(size)

        return sorted(set(resolved))

    @staticmethod
    def _window_starts(n_samples: int, window_size: int, stride: int) -> list[int]:
        if window_size >= n_samples:
            return [0]

        starts = list(range(0, n_samples - window_size + 1, stride))
        last_start = n_samples - window_size
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _classify_window(
        self,
        signal_window: np.ndarray,
        indices: np.ndarray,
    ) -> _TrendSegment | None:
        if len(signal_window) < self.min_window:
            return None

        start_minute = int(indices[0])
        end_minute = int(indices[-1])
        span = end_minute - start_minute
        if span < self.min_span:
            return None

        coverage = len(indices) / (span + 1)
        if coverage < self.min_coverage:
            return None

        x = indices.astype(float)
        y = signal_window.astype(float)
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)

        denom = float(np.dot(x_centered, x_centered))
        if denom <= 0:
            return None

        slope = float(np.dot(x_centered, y_centered) / denom)
        fitted = np.mean(y) + slope * x_centered

        ss_tot = float(np.dot(y_centered, y_centered))
        if ss_tot <= 1e-12:
            return None

        ss_res = float(np.dot(y - fitted, y - fitted))
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        delta = slope * span
        scale = float(np.std(y))
        if scale <= 1e-12:
            return None

        effect = abs(delta) / scale
        if effect < self.min_effect or r2 < self.min_r2:
            return None

        direction = "increasing" if slope > 0 else "decreasing"
        return _TrendSegment(
            direction=direction,
            start_minute=start_minute,
            end_minute=end_minute,
            score=effect * r2,
        )

    @staticmethod
    def _split_opposite_overlaps(
        segments: list[_TrendSegment],
        filtered_signal: np.ndarray,
        indices: np.ndarray,
    ) -> list[_TrendSegment]:
        ordered = sorted(segments, key=lambda s: (s.start_minute, s.end_minute))
        for i, earlier in enumerate(ordered):
            for later in ordered[i + 1:]:
                if earlier.direction == later.direction:
                    continue
                overlap_start = max(earlier.start_minute, later.start_minute)
                overlap_end = min(earlier.end_minute, later.end_minute)
                if overlap_start >= overlap_end:
                    continue
                mask = (indices >= overlap_start) & (indices <= overlap_end)
                if not mask.any():
                    continue
                overlap_values = filtered_signal[mask]
                overlap_indices = indices[mask]
                if earlier.direction == "increasing":
                    split = int(overlap_indices[int(np.argmax(overlap_values))])
                else:
                    split = int(overlap_indices[int(np.argmin(overlap_values))])
                # Score stays at the pre-split value — we do not refit the shortened segments here.
                earlier.end_minute = split
                later.start_minute = split
        return ordered

    @staticmethod
    def _resolve_containment(segments: list[_TrendSegment]) -> list[_TrendSegment]:
        drop: set[int] = set()
        for i, outer in enumerate(segments):
            if i in drop:
                continue
            for j, inner in enumerate(segments):
                if j <= i or j in drop or inner.direction == outer.direction:
                    continue
                if outer.start_minute <= inner.start_minute and inner.end_minute <= outer.end_minute:
                    drop.add(j)
                elif inner.start_minute <= outer.start_minute and outer.end_minute <= inner.end_minute:
                    drop.add(i)
                    break
        return [s for i, s in enumerate(segments) if i not in drop]

    def _merge_segments(self, segments: list[_TrendSegment]) -> list[_TrendSegment]:
        if not segments:
            return []

        segments.sort(key=lambda s: (s.start_minute, s.end_minute, -s.score))
        merged: list[_TrendSegment] = []

        for segment in segments:
            if not merged:
                merged.append(segment)
                continue

            prev = merged[-1]
            if (
                segment.direction == prev.direction
                and segment.start_minute <= prev.end_minute + self.merge_gap
            ):
                prev.end_minute = max(prev.end_minute, segment.end_minute)
                prev.score = max(prev.score, segment.score)
                continue

            merged.append(segment)

        return merged
