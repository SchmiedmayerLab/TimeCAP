#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Signal 6: Weekly exercise volume (W — hourly data).

Fires when any workout channel is active. Totals active hours × 60 as a
weekly exercise-minutes estimate. AHA/WHO: >= 150 min/week moderate activity.
"""
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from clinical.constants import ACTIVITY_CHANNELS
from clinical._base import ClinicalSynthesizer
from timef.schema import Annotation, Recording

AHA_THRESHOLD_MIN = 150   # minutes/week


class ExerciseVolumeSynthesizer(ClinicalSynthesizer):
    """Signal 6 — weekly exercise volume vs. AHA 150-min guideline."""

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        n_time = row.values.shape[1]
        any_active = np.zeros(n_time, dtype=bool)
        channel_idxs: list[int] = []
        for ch in ACTIVITY_CHANNELS:
            try:
                idx = row.channel_names.index(ch)
            except ValueError:
                continue
            vals = np.asarray(row.values[idx], dtype=float)
            ch_active = np.isfinite(vals) & (vals > 0)
            if ch_active.any():
                any_active |= ch_active
                channel_idxs.append(idx)

        active_hours = int(np.sum(any_active))
        if active_hours == 0:
            return []

        # Each time unit is one hour for weekly data
        exercise_min = active_hours * 60
        if exercise_min >= AHA_THRESHOLD_MIN:
            qualifier = f"meets AHA guideline (≥{AHA_THRESHOLD_MIN} min/week)"
        else:
            qualifier = f"below AHA guideline ({AHA_THRESHOLD_MIN} min/week)"

        text = (
            f"Weekly exercise volume: ~{exercise_min} minutes across {active_hours} active hours — "
            f"{qualifier}."
        )
        return [Annotation(
            caption_type="cross_channel",
            text=text,
            channel_idxs=tuple(channel_idxs),
            window=None,
            label="exercise_volume",
        )]
