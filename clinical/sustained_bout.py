#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Signal 17: Sustained activity duration.

Fires when a bout of >= min_cadence steps/min lasts for >= min_duration consecutive minutes.
C2: "if they can never maintain activity for more than 10 minutes at a time, that's a flag."
"""
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from clinical.constants import WATCH_STEP_CHANNEL
from clinical._base import ClinicalSynthesizer
from clinical._helper import contiguous_windows, index_or_none
from timef.schema import Annotation, Recording

MIN_CADENCE = 100   # steps/min — Tudor-Locke moderate intensity threshold
MIN_DURATION = 10   # minutes — C2's "10 minutes at a time" marker


class SustainedBoutSynthesizer(ClinicalSynthesizer):
    """Signal 17 — longest sustained activity bout (cardiovascular reserve proxy)."""

    def __init__(self, min_cadence: float = MIN_CADENCE, min_duration: int = MIN_DURATION):
        self.min_cadence = min_cadence
        self.min_duration = min_duration

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        measured = self._active(row)
        if measured is None:
            return []

        step_idx, steps, active = measured
        bouts = contiguous_windows(active, min_duration=self.min_duration)
        if not bouts:
            return []

        # Report the longest bout only (the key marker, per project_plan)
        longest_start, longest_end = max(bouts, key=lambda b: b[1] - b[0])
        duration = longest_end - longest_start
        mean_cadence = float(np.nanmean(steps[longest_start:longest_end]))
        text = (
            f"Longest sustained activity bout: {duration} minutes "
            f"(minutes {longest_start}–{longest_end}), "
            f"mean cadence {mean_cadence:.0f} steps/min. "
            f"Total bouts ≥{self.min_duration} min: {len(bouts)}."
        )
        return [Annotation(
            caption_type="cross_channel",
            text=text,
            channel_idxs=(step_idx,),
            window=(longest_start, longest_end),
            label="sustained_bout",
        )]

    def features(self, row: Recording) -> dict[str, float]:
        measured = self._active(row)
        if measured is None:
            return {"longest_bout_min": np.nan, "bouts": np.nan}
        # Longest run of any length, so days without a qualifying bout keep their value.
        runs = contiguous_windows(measured[2], min_duration=1)
        return {
            "longest_bout_min": float(max((end - start for start, end in runs), default=0)),
            "bouts": float(sum(1 for start, end in runs if end - start >= self.min_duration)),
        }

    def _active(self, row: Recording) -> tuple[int, np.ndarray, np.ndarray] | None:
        """(step_idx, steps, minutes at or above min_cadence), or None without step data."""
        step_idx = index_or_none(row, WATCH_STEP_CHANNEL)
        if step_idx is None:
            return None

        steps = np.asarray(row.values[step_idx], dtype=float)
        if not np.isfinite(steps).any():
            return None
        return step_idx, steps, np.isfinite(steps) & (steps >= self.min_cadence)
