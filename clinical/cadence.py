#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Signal 18: Step rate / cadence.

Fires when any meaningful movement is detected (max cadence > min_cadence).
Reports peak cadence; flags running (>=130) and low-cadence (<=30) ends.
C2: "if you never get your step rate above ~30 steps/min, that might be Parkinson's."
"""
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from clinical.constants import WATCH_STEP_CHANNEL
from clinical._base import ClinicalSynthesizer
from clinical._helper import index_or_none
from timef.schema import Annotation, Recording

MIN_CADENCE_TO_FIRE = 20      # steps/min — any real purposeful movement
RUNNING_THRESHOLD = 130       # steps/min — Tudor-Locke vigorous / running
LOW_CADENCE_THRESHOLD = 30    # steps/min — C2's neurological concern cutoff


class CadenceSynthesizer(ClinicalSynthesizer):
    """Signal 18 — peak step rate as fitness / neurological indicator."""

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        step_idx = index_or_none(row, WATCH_STEP_CHANNEL)
        if step_idx is None:
            return []

        steps = np.asarray(row.values[step_idx], dtype=float)
        valid = np.isfinite(steps) & (steps > 0)
        if not valid.any():
            return []

        max_cadence = float(np.max(steps[valid]))
        if max_cadence < MIN_CADENCE_TO_FIRE:
            return []

        peak_minute = int(np.argmax(steps))
        if max_cadence >= RUNNING_THRESHOLD:
            qualifier = f"reached running cadence (≥{RUNNING_THRESHOLD} steps/min) — cardiovascular reserve present"
        elif max_cadence <= LOW_CADENCE_THRESHOLD:
            qualifier = f"never exceeded {LOW_CADENCE_THRESHOLD} steps/min — consider neurological or severe deconditioning"
        else:
            qualifier = "moderate walking cadence"

        text = (
            f"Peak cadence: {max_cadence:.0f} steps/min at minute {peak_minute} — {qualifier}."
        )
        return [Annotation(
            caption_type="cross_channel",
            text=text,
            channel_idxs=(step_idx,),
            window=None,
            label="cadence",
        )]
