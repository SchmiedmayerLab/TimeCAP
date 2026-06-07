#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Signal 16: Peak-to-minimum HR spread (dynamic range).

Fires whenever HR data exists. Reports daily max - min HR.
C2: "a relatively high minimum and a low peak is not a good sign."
Compressed range (high min + low peak together) = reduced cardiac reserve.
Not HRV — just the absolute range, derivable from minute-level HR.
"""
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from clinical.constants import WATCH_HR_CHANNEL
from clinical._base import ClinicalSynthesizer
from clinical._helper import index_or_none
from timef.schema import Annotation, Recording

COMPRESSED_MAX = 120   # bpm — peak this low with a high min is concerning
COMPRESSED_MIN = 60    # bpm — min this high is a warning sign


class HRSpreadSynthesizer(ClinicalSynthesizer):
    """Signal 16 — daily HR dynamic range (peak-to-min spread)."""

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        hr_idx = index_or_none(row, WATCH_HR_CHANNEL)
        if hr_idx is None:
            return []

        hr = np.asarray(row.values[hr_idx], dtype=float)
        valid = np.isfinite(hr) & (hr > 0)
        if not valid.any():
            return []

        hr_valid = hr[valid]
        max_hr = float(np.max(hr_valid))
        min_hr = float(np.min(hr_valid))
        spread = max_hr - min_hr

        compressed = (min_hr >= COMPRESSED_MIN) and (max_hr <= COMPRESSED_MAX)
        qualifier = (
            f"compressed range — high minimum + low peak is an unfavorable cardiovascular marker"
            if compressed else
            f"normal range"
        )

        text = (
            f"Daily HR spread: {min_hr:.0f}–{max_hr:.0f} bpm "
            f"(range {spread:.0f} bpm) — {qualifier}."
        )
        return [Annotation(
            caption_type="cross_channel",
            text=text,
            channel_idxs=(hr_idx,),
            window=None,
            label="hr_spread",
        )]
