#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np

from detectors.gap import GapDetector
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
        annotations: list[Annotation] = []

        for group_name, group_channels in self.config.groups.items():
            if group_name != "watch_device":
                continue

            channel_indices = tuple(
                idx for idx, name in enumerate(row.channel_names)
                if name in group_channels
            )
            if len(channel_indices) < self.min_overlapping_channels:
                continue

            eligible = tuple(
                idx for idx in channel_indices
                if any(isinstance(detector, GapDetector) for detector in self.config.detectors.get(row.channel_names[idx], ()))
            )
            if len(eligible) < self.min_overlapping_channels:
                continue

            overlap_windows = _overlap_windows(
                row,
                eligible,
                self.config,
                self.min_overlapping_channels,
                group_name,
                self.min_nonwear_duration,
            )
            if not overlap_windows:
                continue

            device_name = _device_label(group_name)
            for start_minute, end_minute in overlap_windows:
                annotations.append(
                    Annotation(
                        caption_type=self.caption_type,
                        text=f"Potential {device_name} non-wear from minute {start_minute} to {end_minute}.",
                        channel_idxs=channel_indices,
                        window=(start_minute, end_minute),
                        label=group_name,
                    )
                )

        return annotations


def _overlap_windows(
    row: Recording,
    channel_indices: tuple[int, ...],
    config: ChannelConfig,
    min_overlapping_channels: int,
    group_name: str,
    min_nonwear_duration: int,
) -> list[tuple[int, int]]:
    if row.values.shape[1] == 0:
        return []

    channel_masks: dict[int, np.ndarray] = {}
    for idx in channel_indices:
        signal = row.signal(idx)
        channel_mask = np.zeros(row.values.shape[1], dtype=bool)
        for detector in config.detectors.get(signal.name, ()):
            if not isinstance(detector, GapDetector):
                continue
            for result in detector.detect(signal.data):
                channel_mask[result.start_minute:result.end_minute + 1] = True
        channel_masks[idx] = channel_mask

    coverage = np.zeros(row.values.shape[1], dtype=np.int16)
    for channel_mask in channel_masks.values():
        coverage += channel_mask.astype(np.int16)

    if group_name == "watch_device":
        hr_index = _heart_rate_channel_index(row, channel_indices)
        active_energy_index = _active_energy_channel_index(row, channel_indices)
        if hr_index is None or active_energy_index is None:
            return []
        hr_mask = channel_masks.get(hr_index)
        active_energy_mask = channel_masks.get(active_energy_index)
        if hr_mask is None or active_energy_mask is None:
            return []
        overlap_mask = hr_mask & active_energy_mask
    else:
        overlap_mask = coverage >= min_overlapping_channels

    if not overlap_mask.any():
        return []

    padded = np.concatenate(([False], overlap_mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    windows = [
        (int(start), int(end))
        for start, end in zip(starts.tolist(), ends.tolist())
        if end - start + 1 >= min_nonwear_duration
    ]
    return windows


def _device_label(group_name: str) -> str:
    device_name = group_name.removesuffix("_device").replace("_", " ").strip()
    if device_name.lower() == "watch":
        return "Apple Watch"
    return device_name.title()


def _heart_rate_channel_index(row: Recording, channel_indices: tuple[int, ...]) -> int | None:
    for idx in channel_indices:
        if row.channel_names[idx].endswith("HKQuantityTypeIdentifierHeartRate") or "heart_rate" in row.channel_names[idx]:
            return idx
    return None


def _active_energy_channel_index(row: Recording, channel_indices: tuple[int, ...]) -> int | None:
    for idx in channel_indices:
        if row.channel_names[idx].endswith("HKQuantityTypeIdentifierActiveEnergyBurned") or "active_energy" in row.channel_names[idx]:
            return idx
    return None
