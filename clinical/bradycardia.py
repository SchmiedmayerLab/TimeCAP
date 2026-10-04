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
from clinical._base import ClinicalSynthesizer
from clinical._helper import sleep_hr_windows
from timef.schema import Annotation, Recording

# 35 bpm: the <30 bpm cutoff never fired in this cohort (clinical audit). Relaxed to 35
# to catch genuine low-30s dips while staying below typical deep-sleep HR (~38-42 bpm),
# so it does not carpet-flag healthy deep sleepers.
BRADYCARDIA_THRESHOLD = 35.0
MIN_SLEEP_WINDOW = 5  # minutes


class NocturnalBradycardiaSynthesizer(ClinicalSynthesizer):
    """Signal 12 — transient sub-35 bpm HR during sleep (vagal tone of apneic events)."""

    def __init__(self, threshold: float = BRADYCARDIA_THRESHOLD):
        self.threshold = threshold

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        measured = sleep_hr_windows(row, MIN_SLEEP_WINDOW)
        if measured is None:
            return []

        sleep_idx, hr_idx, windows = measured
        results: list[Annotation] = []
        for start, end, hr_window, valid in windows:
            min_hr = float(np.min(hr_window[valid]))
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

    def features(self, row: Recording) -> dict[str, float]:
        measured = sleep_hr_windows(row, MIN_SLEEP_WINDOW)
        if measured is None or not measured[2]:
            return {"min_hr": np.nan, "minutes_below": np.nan}
        hr_valid = np.concatenate([hr_window[valid] for _, _, hr_window, valid in measured[2]])
        return {
            "min_hr": float(np.min(hr_valid)),
            "minutes_below": float(np.sum(hr_valid < self.threshold)),
        }
