#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Signal 21: Circadian HR rhythm / nocturnal HR dipping.

Fires when both daytime resting HR (steps≈0, 08:00–20:00) and overnight HR
(sleep window or 00:00–06:00) are computable. Reports dipping %.
Non-dipping (<10% drop) is an independent hypertension / CVD risk marker.
"""
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from clinical.constants import WATCH_HR_CHANNEL, WATCH_STEP_CHANNEL
from clinical._base import ClinicalSynthesizer
from clinical._helper import index_or_none
from timef.schema import Annotation, Recording

DAYTIME_START = 480   # minute 480 = 08:00
DAYTIME_END = 1200    # minute 1200 = 20:00
OVERNIGHT_END = 360   # minute 360 = 06:00 (use 00:00–06:00 as fallback overnight window)
REST_STEP_THRESHOLD = 5   # steps/min — "resting" = low movement


class CircadianDippingSynthesizer(ClinicalSynthesizer):
    """Signal 21 — nocturnal HR dipping (non-dipping = hypertension/CVD risk)."""

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        measured = self._measure(row)
        if measured is None:
            return []

        mean_day, mean_night, dip_pct, channel_idxs = measured
        if dip_pct < 0:
            qualifier = f"reverse dipper (night HR {abs(dip_pct):.1f}% above day) — strong risk marker"
        elif dip_pct < 10:
            qualifier = f"non-dipper (<10% nocturnal drop) — independent hypertension risk"
        else:
            qualifier = f"normal dipper (≥10% nocturnal drop)"

        text = (
            f"Circadian HR: daytime resting {mean_day:.0f} bpm, "
            f"overnight {mean_night:.0f} bpm, "
            f"dip {dip_pct:.1f}% — {qualifier}."
        )
        is_flag = dip_pct < 10  # non-dipper or reverse dipper
        return [Annotation(
            caption_type="cross_channel",
            text=text,
            channel_idxs=channel_idxs,
            window=None,
            label="circadian_dipping:flag" if is_flag else "circadian_dipping",
        )]

    def features(self, row: Recording) -> dict[str, float]:
        measured = self._measure(row)
        if measured is None:
            return {"day_rest_hr": np.nan, "night_hr": np.nan, "dip_pct": np.nan}
        mean_day, mean_night, dip_pct, _ = measured
        return {"day_rest_hr": mean_day, "night_hr": mean_night, "dip_pct": dip_pct}

    def _measure(self, row: Recording) -> tuple[float, float, float, tuple[int, ...]] | None:
        """(daytime resting HR, overnight HR, dip %, channel_idxs), or None if not computable."""
        hr_idx = index_or_none(row, WATCH_HR_CHANNEL)
        step_idx = index_or_none(row, WATCH_STEP_CHANNEL)
        if hr_idx is None:
            return None

        hr = np.asarray(row.values[hr_idx], dtype=float)
        steps = np.asarray(row.values[step_idx], dtype=float) if step_idx is not None else None

        # Daytime resting HR: minutes 480–1200 where steps ≈ 0
        daytime_mask = np.zeros(len(hr), dtype=bool)
        daytime_mask[DAYTIME_START:DAYTIME_END] = True
        if steps is not None:
            rest_mask = np.isfinite(steps) & (steps < REST_STEP_THRESHOLD)
            daytime_rest = daytime_mask & rest_mask
        else:
            daytime_rest = daytime_mask

        hr_daytime = hr[daytime_rest]
        hr_daytime = hr_daytime[np.isfinite(hr_daytime) & (hr_daytime > 0)]
        if len(hr_daytime) < 10:
            return None
        mean_day = float(np.mean(hr_daytime))

        # Overnight HR: prefer sleep:asleep window, fall back to 00:00–06:00
        overnight_hr = self._overnight_hr(row, hr)
        if overnight_hr is None or len(overnight_hr) < 5:
            return None
        mean_night = float(np.mean(overnight_hr))

        dip_pct = (mean_day - mean_night) / mean_day * 100.0
        channel_idxs = (hr_idx,) if step_idx is None else (hr_idx, step_idx)
        return mean_day, mean_night, dip_pct, channel_idxs

    @staticmethod
    def _overnight_hr(row: Recording, hr: np.ndarray) -> np.ndarray | None:
        try:
            sleep_idx = row.channel_names.index("sleep:asleep")
            sleep = np.asarray(row.values[sleep_idx], dtype=float)
            sleep_mask = np.isfinite(sleep) & (sleep > 0)
            if sleep_mask.any():
                hr_sleep = hr[sleep_mask]
                hr_sleep = hr_sleep[np.isfinite(hr_sleep) & (hr_sleep > 0)]
                if len(hr_sleep) >= 5:
                    return hr_sleep
        except ValueError:
            pass
        # Fallback: 00:00–06:00
        hr_early = hr[:OVERNIGHT_END]
        hr_valid = hr_early[np.isfinite(hr_early) & (hr_early > 0)]
        return hr_valid if len(hr_valid) >= 5 else None
