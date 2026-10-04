#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Signal 7: High HR asleep / low HR exercising (HR inversion).

Fires when mean HR during sleep > mean HR during exercise on the same day.
C1: "this is weird — their heart rate is higher asleep than exercising."
Rate-controlling medication or CVD is the leading hypothesis (but not a diagnosis).
"""
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from clinical.constants import ACTIVITY_CHANNELS, WATCH_HR_CHANNEL
from clinical._base import ClinicalSynthesizer
from clinical._helper import contiguous_windows, index_or_none
from timef.schema import Annotation, Recording

MIN_SLEEP_DURATION = 5   # minutes
MIN_WORKOUT_DURATION = 5  # minutes


class HRInversionSynthesizer(ClinicalSynthesizer):
    """Signal 7 — sleep HR exceeds exercise HR (pathological inversion)."""

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        measured = self._measure(row)
        if measured is None:
            return []

        hr_idx, sleep_hr, exercise_hr = measured
        if sleep_hr <= exercise_hr:
            return []

        text = (
            f"HR inversion: mean HR during sleep {sleep_hr:.0f} bpm "
            f"> mean HR during exercise {exercise_hr:.0f} bpm. "
            f"Blunted exercise response with preserved nocturnal capacity — "
            f"consider rate-controlling medication or cardiovascular disease."
        )
        return [Annotation(
            caption_type="cross_channel",
            text=text,
            channel_idxs=(hr_idx,),
            window=None,
            label="hr_inversion:flag",
        )]

    def features(self, row: Recording) -> dict[str, float]:
        measured = self._measure(row)
        if measured is None:
            return {"sleep_hr": np.nan, "exercise_hr": np.nan, "sleep_minus_exercise": np.nan}
        _, sleep_hr, exercise_hr = measured
        return {
            "sleep_hr": sleep_hr,
            "exercise_hr": exercise_hr,
            "sleep_minus_exercise": sleep_hr - exercise_hr,
        }

    def _measure(self, row: Recording) -> tuple[int, float, float] | None:
        """(hr_idx, mean sleep HR, mean exercise HR), or None unless both are measurable."""
        hr_idx = index_or_none(row, WATCH_HR_CHANNEL)
        if hr_idx is None:
            return None

        hr = np.asarray(row.values[hr_idx], dtype=float)

        sleep_hr = self._mean_hr_in_window(row, hr, "sleep:asleep", MIN_SLEEP_DURATION)
        if sleep_hr is None:
            return None

        exercise_hr = self._mean_exercise_hr(row, hr)
        if exercise_hr is None:
            return None
        return hr_idx, sleep_hr, exercise_hr

    @staticmethod
    def _mean_hr_in_window(
        row: Recording, hr: np.ndarray, channel_name: str, min_duration: int
    ) -> float | None:
        try:
            idx = row.channel_names.index(channel_name)
        except ValueError:
            return None
        vals = np.asarray(row.values[idx], dtype=float)
        mask = np.isfinite(vals) & (vals > 0)
        bouts = contiguous_windows(mask, min_duration=min_duration)
        if not bouts:
            return None
        all_hr = np.concatenate([hr[s:e] for s, e in bouts])
        valid = all_hr[np.isfinite(all_hr) & (all_hr > 0)]
        return float(np.mean(valid)) if len(valid) > 0 else None

    @staticmethod
    def _mean_exercise_hr(row: Recording, hr: np.ndarray) -> float | None:
        n = row.values.shape[1]
        mask = np.zeros(n, dtype=bool)
        for ch in ACTIVITY_CHANNELS:
            try:
                idx = row.channel_names.index(ch)
            except ValueError:
                continue
            vals = np.asarray(row.values[idx], dtype=float)
            mask |= np.isfinite(vals) & (vals > 0)
        bouts = contiguous_windows(mask, min_duration=MIN_WORKOUT_DURATION)
        if not bouts:
            return None
        all_hr = np.concatenate([hr[s:e] for s, e in bouts])
        valid = all_hr[np.isfinite(all_hr) & (all_hr > 0)]
        return float(np.mean(valid)) if len(valid) > 0 else None
