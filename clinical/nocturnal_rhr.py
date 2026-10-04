#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Signal 20: Overnight resting HR (per-night, for COVID / infection trend).

Fires when a sleep window with HR data exists. Returns the per-night mean
overnight RHR — the building block for the NightSignal rising-trend detector.
Across the week (7 nightly captions) a rising trend is the infection signal.
"""
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from clinical._base import ClinicalSynthesizer
from clinical._helper import sleep_hr_windows
from timef.schema import Annotation, Recording

MIN_NIGHT_WINDOW = 30  # minutes


class NocturnalRHRSynthesizer(ClinicalSynthesizer):
    """Signal 20 — per-night overnight RHR (infection / COVID trend baseline)."""

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        measured = sleep_hr_windows(row, MIN_NIGHT_WINDOW)
        if measured is None:
            return []

        sleep_idx, hr_idx, windows = measured
        results: list[Annotation] = []
        for start, end, hr_window, valid in windows:
            mean_rhr = float(np.mean(hr_window[valid]))
            text = (
                f"Overnight resting HR (minutes {start}–{end}): "
                f"{mean_rhr:.0f} bpm. "
                f"(Context: a sustained rise ≥3–4 bpm above personal baseline over successive nights "
                f"is the pre-symptomatic infection signature per NightSignal.)"
            )
            results.append(Annotation(
                caption_type="cross_channel",
                text=text,
                channel_idxs=(sleep_idx, hr_idx),
                window=(start, end),
                label="nocturnal_rhr:flag" if mean_rhr > 80 else "nocturnal_rhr",
            ))
        return results

    def features(self, row: Recording) -> dict[str, float]:
        measured = sleep_hr_windows(row, MIN_NIGHT_WINDOW)
        if measured is None or not measured[2]:
            return {"mean_hr": np.nan}
        hr_valid = np.concatenate([hr_window[valid] for _, _, hr_window, valid in measured[2]])
        return {"mean_hr": float(np.mean(hr_valid))}
