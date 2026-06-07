#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Signal 24: Sedentary burden (prolonged inactivity bouts).

Fires when any unbroken sedentary run >= min_duration minutes occurs during waking hours
(minutes 420–1320, i.e., 07:00–22:00). Steps < 5/min = sedentary.
"""
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from clinical.constants import WATCH_STEP_CHANNEL
from clinical._base import ClinicalSynthesizer
from clinical._helper import contiguous_windows, index_or_none
from timef.schema import Annotation, Recording

WAKING_START = 420    # minute 420 = 07:00
WAKING_END = 1320     # minute 1320 = 22:00
SEDENTARY_THRESHOLD = 5   # steps/min
MIN_DURATION = 60         # minutes — Biswas et al. threshold


class SedentaryBurdenSynthesizer(ClinicalSynthesizer):
    """Signal 24 — longest unbroken sedentary run during waking hours."""

    def __init__(self, min_duration: int = MIN_DURATION):
        self.min_duration = min_duration

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        step_idx = index_or_none(row, WATCH_STEP_CHANNEL)
        if step_idx is None:
            return []

        steps = np.asarray(row.values[step_idx], dtype=float)
        # Waking hours mask; outside waking = not sedentary (could be sleep)
        waking = np.zeros(len(steps), dtype=bool)
        waking[WAKING_START:WAKING_END] = True

        sedentary = waking & (np.isfinite(steps) & (steps < SEDENTARY_THRESHOLD))
        bouts = contiguous_windows(sedentary, min_duration=self.min_duration)
        if not bouts:
            return []

        longest_start, longest_end = max(bouts, key=lambda b: b[1] - b[0])
        duration = longest_end - longest_start
        text = (
            f"Longest unbroken sedentary period during waking hours: {duration} minutes "
            f"(minutes {longest_start}–{longest_end}). "
            f"Total sedentary bouts ≥{self.min_duration} min: {len(bouts)}."
        )
        return [Annotation(
            caption_type="cross_channel",
            text=text,
            channel_idxs=(step_idx,),
            window=(longest_start, longest_end),
            label="sedentary_burden:flag",
        )]
