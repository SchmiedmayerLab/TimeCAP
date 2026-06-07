#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Signal 3: Resting / baseline HR.

Fires when a sleep window exists and HR data is present during it.
Captions the overnight sleeping HR (mean, min, max).
"""
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from clinical.constants import WATCH_HR_CHANNEL
from clinical._base import ClinicalSynthesizer
from clinical._helper import contiguous_windows, index_or_none
from timef.schema import Annotation, Recording


class RestingHRSynthesizer(ClinicalSynthesizer):
    """Signal 3 — overnight resting HR level."""

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        try:
            sleep_idx = row.channel_names.index("sleep:asleep")
        except ValueError:
            return []

        hr_idx = index_or_none(row, WATCH_HR_CHANNEL)
        if hr_idx is None:
            return []

        sleep = np.asarray(row.values[sleep_idx], dtype=float)
        sleep_mask = np.isfinite(sleep) & (sleep > 0)
        if not sleep_mask.any():
            return []

        hr = np.asarray(row.values[hr_idx], dtype=float)
        results: list[Annotation] = []
        for start, end in contiguous_windows(sleep_mask, min_duration=5):
            hr_window = hr[start:end]
            valid = np.isfinite(hr_window) & (hr_window > 0)
            if not valid.any():
                continue
            hr_valid = hr_window[valid]
            mean_hr = float(np.mean(hr_valid))
            min_hr = float(np.min(hr_valid))
            max_hr = float(np.max(hr_valid))
            text = (
                f"Resting HR during sleep (minutes {start}–{end}): "
                f"mean {mean_hr:.0f} bpm, range {min_hr:.0f}–{max_hr:.0f} bpm."
            )
            results.append(Annotation(
                caption_type="cross_channel",
                text=text,
                channel_idxs=(sleep_idx, hr_idx),
                window=(start, end),
                label="resting_hr:flag" if mean_hr > 80 else "resting_hr",
            ))
        return results
