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
from clinical.constants import WATCH_HR_CHANNEL
from clinical._base import ClinicalSynthesizer
from clinical._helper import contiguous_windows, index_or_none
from timef.schema import Annotation, Recording


class NocturnalRHRSynthesizer(ClinicalSynthesizer):
    """Signal 20 — per-night overnight RHR (infection / COVID trend baseline)."""

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
        for start, end in contiguous_windows(sleep_mask, min_duration=30):
            hr_window = hr[start:end]
            valid = np.isfinite(hr_window) & (hr_window > 0)
            if not valid.any():
                continue
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
