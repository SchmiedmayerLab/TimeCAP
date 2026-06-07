#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Signal 1: HR recovery after exercise.

Fires when a workout window exists with HR data, and HR drops detectably after
the workout ends. Measures minutes until HR returns within +recovery_margin bpm
of the pre-workout (sleeping) baseline.
Note: HRR-60 cutoffs from graded exercise tests don't transfer to free-living
data — use direction only (faster = healthier), not bpm cutoffs.
"""
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from clinical.constants import ACTIVITY_CHANNELS, WATCH_HR_CHANNEL
from clinical._base import ClinicalSynthesizer
from clinical._helper import contiguous_windows, index_or_none
from timef.schema import Annotation, Recording

RECOVERY_MARGIN = 10    # bpm — "returned to baseline" = within +10 bpm
POST_WORKOUT_WINDOW = 60  # minutes to search for recovery after workout ends
MIN_WORKOUT_DURATION = 5  # minutes


class HRRecoverySynthesizer(ClinicalSynthesizer):
    """Signal 1 — post-exercise HR recovery speed (vagal reactivation proxy)."""

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        hr_idx = index_or_none(row, WATCH_HR_CHANNEL)
        if hr_idx is None:
            return []

        hr = np.asarray(row.values[hr_idx], dtype=float)
        baseline = self._resting_baseline(row, hr)
        if baseline is None:
            return []

        workout_mask = self._workout_mask(row)
        bouts = contiguous_windows(workout_mask, min_duration=MIN_WORKOUT_DURATION)
        if not bouts:
            return []

        results: list[Annotation] = []
        for start, end in bouts:
            hr_during = hr[start:end]
            valid = np.isfinite(hr_during) & (hr_during > 0)
            if not valid.any():
                continue
            peak_hr = float(np.max(hr_during[valid]))
            if peak_hr < baseline + 20:
                # HR didn't rise enough above baseline to call it an exercise response
                continue

            # Search for recovery in the post-workout window
            post_end = min(end + POST_WORKOUT_WINDOW, len(hr))
            recovery_min = None
            for t in range(end, post_end):
                h = hr[t]
                if np.isfinite(h) and h > 0 and h <= baseline + RECOVERY_MARGIN:
                    recovery_min = t - end
                    break

            duration = end - start
            if recovery_min is not None:
                qualifier = f"HR recovered in {recovery_min} min"
            else:
                qualifier = f"HR had not recovered within {POST_WORKOUT_WINDOW} min (prolonged elevation)"

            text = (
                f"Exercise bout (minutes {start}–{end}, {duration} min): "
                f"peak HR {peak_hr:.0f} bpm vs baseline {baseline:.0f} bpm — {qualifier}."
            )
            results.append(Annotation(
                caption_type="cross_channel",
                text=text,
                channel_idxs=(hr_idx,),
                window=(start, end),
                label="hr_recovery",
            ))
        return results

    @staticmethod
    def _resting_baseline(row: Recording, hr: np.ndarray) -> float | None:
        try:
            sleep_idx = row.channel_names.index("sleep:asleep")
            sleep = np.asarray(row.values[sleep_idx], dtype=float)
            sleep_mask = np.isfinite(sleep) & (sleep > 0)
            if sleep_mask.any():
                hr_sleep = hr[sleep_mask]
                hr_sleep = hr_sleep[np.isfinite(hr_sleep) & (hr_sleep > 0)]
                if len(hr_sleep) >= 5:
                    return float(np.mean(hr_sleep))
        except ValueError:
            pass
        # Fallback: 25th percentile of all positive HR values
        valid = hr[np.isfinite(hr) & (hr > 0)]
        if len(valid) < 10:
            return None
        return float(np.percentile(valid, 25))

    @staticmethod
    def _workout_mask(row: Recording) -> np.ndarray:
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
