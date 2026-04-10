#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np

from extractors import CaptionExtractor, ChannelConfig
from timef.schema import Annotation, Recording


class DeviceWearExtractor(CaptionExtractor):
    caption_type = "watch_wear"

    def __init__(
        self,
        config: ChannelConfig,
        min_overlapping_channels: int = 2,
        min_nonwear_duration: int = 30,
    ):
        super().__init__(config)
        self.min_overlapping_channels = min_overlapping_channels
        self.min_nonwear_duration = min_nonwear_duration

    def extract(self, row: Recording) -> list[Annotation]:
        group_channels = self.config.groups.get("watch_device")
        if not group_channels:
            return []

        channel_indices = tuple(
            idx for idx, name in enumerate(row.channel_names)
            if name in group_channels
        )
        if len(channel_indices) < self.min_overlapping_channels:
            return []

        hr_index = _heart_rate_channel_index(row, channel_indices)
        active_energy_index = _active_energy_channel_index(row, channel_indices)
        if hr_index is None or active_energy_index is None:
            return []

        hr_gap_mask = _gap_mask(row.signal(hr_index).data)
        active_energy_gap_mask = _gap_mask(row.signal(active_energy_index).data)
        nonwear_windows = _mask_to_windows(
            hr_gap_mask & active_energy_gap_mask,
            min_duration=self.min_nonwear_duration,
        )
        return [
            Annotation(
                caption_type=self.caption_type,
                text=f"Potential Apple Watch non-wear from minute {start_minute} to {end_minute}.",
                channel_idxs=channel_indices,
                window=(start_minute, end_minute),
                label="watch_device",
            )
            for start_minute, end_minute in nonwear_windows
        ]


def _gap_mask(series: np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    return np.isnan(arr) | (arr == 0)


def _mask_to_windows(mask: np.ndarray, min_duration: int) -> list[tuple[int, int]]:
    if mask.size == 0 or not mask.any():
        return []

    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    return [
        (int(start), int(end))
        for start, end in zip(starts.tolist(), ends.tolist())
        if end - start + 1 >= min_duration
    ]


def _heart_rate_channel_index(row: Recording, channel_indices: tuple[int, ...]) -> int | None:
    for idx in channel_indices:
        name = row.channel_names[idx]
        if name.endswith("HKQuantityTypeIdentifierHeartRate") or "heart_rate" in name:
            return idx
    return None


def _active_energy_channel_index(row: Recording, channel_indices: tuple[int, ...]) -> int | None:
    for idx in channel_indices:
        name = row.channel_names[idx]
        if name.endswith("HKQuantityTypeIdentifierActiveEnergyBurned") or "active_energy" in name:
            return idx
    return None
