#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Signal 14: Nocturnal activity.

Fires when steps are recorded inside the sleep window (sleep:asleep or sleep:inbed).
C2: "why are they registered as climbing a flight in the middle of the night?"
Interpretation: nocturia, late bedtime, or restlessness — nonspecific, hedge heavily.
"""
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from clinical.constants import WATCH_STEP_CHANNEL
from clinical._base import ClinicalSynthesizer
from clinical._helper import contiguous_windows, index_or_none
from timef.schema import Annotation, Recording

MIN_STEPS = 5       # steps/min to count as "activity" (not a wrist artifact)
MIN_DURATION = 1    # minute


class NocturnalActivitySynthesizer(ClinicalSynthesizer):
    """Signal 14 — steps recorded inside the sleep window (nocturia / fragmentation)."""

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        step_idx = index_or_none(row, WATCH_STEP_CHANNEL)
        if step_idx is None:
            return []

        sleep_mask = self._sleep_mask(row)
        if not sleep_mask.any():
            return []

        steps = np.asarray(row.values[step_idx], dtype=float)
        active_during_sleep = sleep_mask & np.isfinite(steps) & (steps >= MIN_STEPS)
        bouts = contiguous_windows(active_during_sleep, min_duration=MIN_DURATION)
        if not bouts:
            return []

        events = []
        for start, end in bouts:
            mean_steps = float(np.nanmean(steps[start:end]))
            events.append(f"minutes {start}–{end} ({mean_steps:.0f} steps/min)")

        text = (
            f"Activity during sleep window: {len(bouts)} event(s) — "
            + ", ".join(events[:3])
            + (f" (+{len(bouts) - 3} more)" if len(bouts) > 3 else "")
            + ". Possible nocturia or sleep fragmentation."
        )
        channel_idxs = (step_idx,)
        return [Annotation(
            caption_type="cross_channel",
            text=text,
            channel_idxs=channel_idxs,
            window=(bouts[0][0], bouts[-1][1]),
            label="nocturnal_activity:flag",
        )]

    @staticmethod
    def _sleep_mask(row: Recording) -> np.ndarray:
        n = row.values.shape[1]
        mask = np.zeros(n, dtype=bool)
        for ch in ("sleep:asleep", "sleep:inbed"):
            try:
                idx = row.channel_names.index(ch)
            except ValueError:
                continue
            vals = np.asarray(row.values[idx], dtype=float)
            mask |= np.isfinite(vals) & (vals > 0)
        return mask
