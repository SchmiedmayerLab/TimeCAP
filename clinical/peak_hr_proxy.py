#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Signal 11: Peak HR as age / fitness proxy (W — hourly data).

Fires whenever HR data exists. Reports the weekly max HR (hourly average peak).
C2: "I see a peak of 112, which to me says they're a pretty old person straight off the bat."
At hourly resolution this underestimates true minute-level peaks, so the number
is a floor, not a ceiling. Compare to 220-age if age is available in labels.
"""
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from clinical.constants import WATCH_HR_CHANNEL
from clinical._base import ClinicalSynthesizer
from clinical._helper import index_or_none
from timef.schema import Annotation, Recording

LOW_PEAK_THRESHOLD = 120   # bpm — C2's informal "pretty old / very inactive" indicator


class PeakHRProxySynthesizer(ClinicalSynthesizer):
    """Signal 11 — weekly max HR as fitness / age proxy (hourly-resolution floor)."""

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        hr_idx = index_or_none(row, WATCH_HR_CHANNEL)
        if hr_idx is None:
            return []

        hr = np.asarray(row.values[hr_idx], dtype=float)
        valid = np.isfinite(hr) & (hr > 0)
        if not valid.any():
            return []

        max_hr = float(np.nanmax(hr))
        peak_hour = int(np.nanargmax(hr))

        qualifier = (
            f"low weekly peak — suggests older or markedly sedentary individual (C2 threshold ~{LOW_PEAK_THRESHOLD} bpm)"
            if max_hr < LOW_PEAK_THRESHOLD else
            f"adequate weekly peak"
        )
        text = (
            f"Weekly peak HR: {max_hr:.0f} bpm (hour {peak_hour}) — {qualifier}. "
            f"Note: hourly resolution underestimates true minute-level peak."
        )
        return [Annotation(
            caption_type="cross_channel",
            text=text,
            channel_idxs=(hr_idx,),
            window=None,
            label="peak_hr_proxy:flag" if max_hr < LOW_PEAK_THRESHOLD else "peak_hr_proxy",
        )]
