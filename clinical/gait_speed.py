#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Signal 22: Gait speed / stride length.

Fires during any workout window (or step bout) where both steps and distance
channels are populated. Stride length = distance (m/min) / steps (steps/min).
Studenski et al. JAMA 2011: gait speed < 0.8 m/s = functional limitation.
"""
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from clinical.constants import (
    ACTIVITY_CHANNELS,
    WATCH_DISTANCE_CHANNEL,
    WATCH_STEP_CHANNEL,
)
from clinical._base import ClinicalSynthesizer
from clinical._helper import contiguous_windows, index_or_none
from timef.schema import Annotation, Recording

MIN_STEPS_PER_MIN = 40    # below this, stride computation is unreliable
MIN_BOUT_DURATION = 3     # minutes


class GaitSpeedSynthesizer(ClinicalSynthesizer):
    """Signal 22 — stride length during walking bouts (gait speed proxy)."""

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        measured = self._measure(row)
        if measured is None:
            return []

        step_idx, dist_idx, bouts = measured
        results: list[Annotation] = []
        for start, end, stride_m, speed_ms in bouts:
            duration = end - start

            if speed_ms < 0.8:
                qualifier = "below functional threshold (< 0.8 m/s)"
            elif speed_ms < 1.0:
                qualifier = "slow (0.8–1.0 m/s)"
            else:
                qualifier = "normal (≥ 1.0 m/s)"

            text = (
                f"Walking bout (minutes {start}–{end}, {duration} min): "
                f"stride {stride_m:.2f} m, speed {speed_ms:.2f} m/s — {qualifier}."
            )
            results.append(Annotation(
                caption_type="cross_channel",
                text=text,
                channel_idxs=(step_idx, dist_idx),
                window=(start, end),
                label="gait_speed:flag" if speed_ms < 0.8 else "gait_speed",
            ))
        return results

    def features(self, row: Recording) -> dict[str, float]:
        measured = self._measure(row)
        if measured is None or not measured[2]:
            return {"speed_mean": np.nan, "speed_min": np.nan, "stride_mean": np.nan, "bouts": np.nan}
        bouts = measured[2]
        speeds = [speed_ms for _, _, _, speed_ms in bouts]
        return {
            "speed_mean": float(np.mean(speeds)),
            "speed_min": float(np.min(speeds)),
            "stride_mean": float(np.mean([stride_m for _, _, stride_m, _ in bouts])),
            "bouts": float(len(bouts)),
        }

    def _measure(
        self, row: Recording
    ) -> tuple[int, int, list[tuple[int, int, float, float]]] | None:
        """(step_idx, dist_idx, [(start, end, stride m, speed m/s), ...]), or None without channels."""
        step_idx = index_or_none(row, WATCH_STEP_CHANNEL)
        dist_idx = index_or_none(row, WATCH_DISTANCE_CHANNEL)
        if step_idx is None or dist_idx is None:
            return None

        # Detect active windows: any workout channel active, OR sustained stepping
        active_mask = self._active_mask(row)
        steps = np.asarray(row.values[step_idx], dtype=float)
        dist = np.asarray(row.values[dist_idx], dtype=float)
        bouts: list[tuple[int, int, float, float]] = []

        for start, end in contiguous_windows(active_mask, min_duration=MIN_BOUT_DURATION):
            s = steps[start:end]
            d = dist[start:end]
            valid = np.isfinite(s) & (s >= MIN_STEPS_PER_MIN) & np.isfinite(d) & (d > 0)
            if not valid.any():
                continue
            mean_steps = float(np.mean(s[valid]))
            mean_dist = float(np.mean(d[valid]))
            # stride length in metres per step; walking speed in m/s = (m/min) / 60
            bouts.append((start, end, mean_dist / mean_steps, mean_dist / 60.0))
        return step_idx, dist_idx, bouts

    @staticmethod
    def _active_mask(row: Recording) -> np.ndarray:
        n = row.values.shape[1]
        mask = np.zeros(n, dtype=bool)
        for ch in ACTIVITY_CHANNELS:
            try:
                idx = row.channel_names.index(ch)
            except ValueError:
                continue
            vals = np.asarray(row.values[idx], dtype=float)
            mask |= np.isfinite(vals) & (vals > 0)
        return mask
