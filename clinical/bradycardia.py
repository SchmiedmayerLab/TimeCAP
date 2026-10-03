#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Signal 12: Nocturnal bradycardia < 35 bpm.

Fires when HR dips below the bradycardia threshold during any sleep window at
minute resolution. Hourly data cannot detect this — minute-level only.
"""
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from clinical.constants import WATCH_HR_CHANNEL
from clinical._base import ClinicalSynthesizer
from clinical._helper import contiguous_windows, index_or_none
from timef.schema import Annotation, Recording

# 35 bpm: the <30 bpm cutoff never fired in this cohort (clinical audit). Relaxed to 35
# to catch genuine low-30s dips while staying below typical deep-sleep HR (~38-42 bpm),
# so it does not carpet-flag healthy deep sleepers.
BRADYCARDIA_THRESHOLD = 35.0


class NocturnalBradycardiaSynthesizer(ClinicalSynthesizer):
    """Signal 12 — transient sub-35 bpm HR during sleep (vagal tone of apneic events)."""

    def __init__(self, threshold: float = BRADYCARDIA_THRESHOLD):
        self.threshold = threshold

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
            min_hr = float(np.min(hr_valid))
            if min_hr >= self.threshold:
                continue
            # Find first minute within the window where HR is below threshold
            below = np.where(valid & (hr_window < self.threshold))[0]
            dip_minute = int(start + below[0])
            text = (
                f"During sleep (minutes {start}–{end}), HR dipped to {min_hr:.0f} bpm "
                f"at minute {dip_minute} — below the {self.threshold:.0f} bpm bradycardia threshold."
            )
            results.append(Annotation(
                caption_type="cross_channel",
                text=text,
                channel_idxs=(sleep_idx, hr_idx),
                window=(start, end),
                label="nocturnal_bradycardia:flag",
            ))
        return results
