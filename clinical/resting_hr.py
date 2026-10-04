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
from clinical._base import ClinicalSynthesizer
from clinical._helper import sleep_hr_windows
from timef.schema import Annotation, Recording

MIN_SLEEP_WINDOW = 5  # minutes


class RestingHRSynthesizer(ClinicalSynthesizer):
    """Signal 3 — overnight resting HR level."""

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        measured = sleep_hr_windows(row, MIN_SLEEP_WINDOW)
        if measured is None:
            return []

        sleep_idx, hr_idx, windows = measured
        results: list[Annotation] = []
        for start, end, hr_window, valid in windows:
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

    def features(self, row: Recording) -> dict[str, float]:
        measured = sleep_hr_windows(row, MIN_SLEEP_WINDOW)
        if measured is None or not measured[2]:
            return {"mean_hr": np.nan, "min_hr": np.nan, "max_hr": np.nan}
        hr_valid = np.concatenate([hr_window[valid] for _, _, hr_window, valid in measured[2]])
        return {
            "mean_hr": float(np.mean(hr_valid)),
            "min_hr": float(np.min(hr_valid)),
            "max_hr": float(np.max(hr_valid)),
        }
