#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
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


class MonotonicTrendDetector(StructuralDetector):
    """Detects monotonic trends, then merges runs across controlled outliers."""

    def __init__(
        self,
        filter_zeros: bool = False,
        min_points: int = 3,
        min_span: int = 1,
        min_length: int = 10,
        max_outliers: int = 3,
        min_change: float = 0.0,
        trim_edge_plateaus: bool = True,
    ) -> None:
        super().__init__(filter_zeros=filter_zeros)
        self.min_points = max(2, min_points)
        self.min_span = max(1, min_span)
        self.min_length = max(1, min_length)
        self.max_outliers = max(0, max_outliers)
        self.min_change = max(0.0, min_change)
        self.trim_edge_plateaus = trim_edge_plateaus

    def _detect(self, filtered_signal: np.ndarray, indices: np.ndarray) -> list[DetectionResult]:
        segments: list[_TrendSegment] = []
        for direction in ("increasing", "decreasing"):
            direction_segments = self._find_monotonic_segments(filtered_signal, indices, direction)
            merged = self._merge_with_outliers(direction_segments, filtered_signal, indices, direction)
            segments.extend(segment for segment in merged if self._passes_minimums(segment, filtered_signal, indices))

        return self._resolve_overlapping_directions(segments)

    def _find_monotonic_segments(
        self,
        signal: np.ndarray,
        indices: np.ndarray,
        direction: str,
    ) -> list[_TrendSegment]:
        if len(signal) < self.min_points:
            return []

        is_ordered = np.greater_equal if direction == "increasing" else np.less_equal
        segments: list[_TrendSegment] = []
        start: int | None = None
        for pos in range(1, len(signal)):
            if is_ordered(signal[pos], signal[pos - 1]):
                if start is None:
                    start = pos - 1
                continue
            if start is not None:
                self._append_segment(segments, direction, start, pos - 1, signal, indices)
                start = None

        if start is not None:
            self._append_segment(segments, direction, start, len(signal) - 1, signal, indices)
        return segments

    def _append_segment(
        self,
        segments: list[_TrendSegment],
        direction: str,
        start_pos: int,
        end_pos: int,
        signal: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        if start_pos > end_pos:
            return
        if self.trim_edge_plateaus:
            start_pos, end_pos = self._trim_edge_plateaus(signal, start_pos, end_pos, direction)

        if end_pos - start_pos + 1 < self.min_points:
            return

        start_minute = int(indices[start_pos])
        end_minute = int(indices[end_pos])
        if end_minute - start_minute < self.min_span:
            return

        delta = float(signal[end_pos] - signal[start_pos])
        if direction == "increasing" and delta <= self.min_change:
            return
        if direction == "decreasing" and -delta <= self.min_change:
            return

        segments.append(
            _TrendSegment(
                direction=direction,
                start_minute=start_minute,
                end_minute=end_minute,
                score=abs(delta),
            )
        )

    @staticmethod
    def _trim_edge_plateaus(signal: np.ndarray, start_pos: int, end_pos: int, direction: str) -> tuple[int, int]:
        if start_pos >= end_pos:
            return start_pos, end_pos

        if direction == "increasing":
            strict_steps = np.flatnonzero(signal[start_pos + 1 : end_pos + 1] > signal[start_pos:end_pos])
        else:
            strict_steps = np.flatnonzero(signal[start_pos + 1 : end_pos + 1] < signal[start_pos:end_pos])

        if len(strict_steps) == 0:
            return start_pos, end_pos

        first_strict_step = start_pos + int(strict_steps[0]) + 1
        last_strict_step = start_pos + int(strict_steps[-1]) + 1
        trimmed_start = first_strict_step - 1
        trimmed_end = last_strict_step
        return trimmed_start, trimmed_end

    def _merge_with_outliers(
        self,
        segments: list[_TrendSegment],
        signal: np.ndarray,
        indices: np.ndarray,
        direction: str,
    ) -> list[_TrendSegment]:
        if not segments:
            return []

        merged = [segments[0]]
        for segment in segments[1:]:
            previous = merged[-1]
            if self._can_merge_gap(previous, segment, signal, indices, direction):
                previous.end_minute = segment.end_minute
                previous.score += segment.score
                continue

            merged.append(segment)

        return merged

    def _can_merge_gap(
        self,
        previous: _TrendSegment,
        segment: _TrendSegment,
        signal: np.ndarray,
        indices: np.ndarray,
        direction: str,
    ) -> bool:
        previous_end_value = float(signal[indices == previous.end_minute][0])
        segment_start_value = float(signal[indices == segment.start_minute][0])
        violates_boundary = (
            segment_start_value < previous_end_value
            if direction == "increasing"
            else segment_start_value > previous_end_value
        )

        gap_mask = (indices > previous.end_minute) & (indices < segment.start_minute)
        if not gap_mask.any() and violates_boundary:
            gap_mask = indices == segment.start_minute

        outlier_count = int(np.count_nonzero(gap_mask))
        if outlier_count > self.max_outliers:
            return False
        if outlier_count == 0:
            return True

        start_value = float(signal[indices == previous.start_minute][0])
        gap_values = signal[gap_mask]
        if direction == "increasing":
            return bool(np.all(gap_values >= start_value))
        return bool(np.all(gap_values <= start_value))

    def _passes_minimums(self, segment: _TrendSegment, signal: np.ndarray, indices: np.ndarray) -> bool:
        mask = (indices >= segment.start_minute) & (indices <= segment.end_minute)
        if int(np.count_nonzero(mask)) < self.min_points:
            return False
        if segment.end_minute - segment.start_minute + 1 < self.min_length:
            return False

        values = signal[mask]
        delta = float(values[-1] - values[0])
        if segment.direction == "increasing":
            return delta > self.min_change
        return -delta > self.min_change

    @staticmethod
    def _resolve_overlapping_directions(segments: list[_TrendSegment]) -> list[DetectionResult]:
        segments.sort(key=lambda segment: (-segment.score, segment.start_minute, segment.end_minute))
        accepted: list[_TrendSegment] = []
        for segment in segments:
            if any(
                segment.start_minute <= accepted_segment.end_minute
                and accepted_segment.start_minute <= segment.end_minute
                for accepted_segment in accepted
            ):
                continue
            accepted.append(segment)

        accepted.sort(key=lambda segment: (segment.start_minute, segment.end_minute))
        return [
            DetectionResult(
                event_type="trend",
                start_minute=segment.start_minute,
                end_minute=segment.end_minute,
                direction=segment.direction,
                score=segment.score,
            )
            for segment in accepted
        ]
