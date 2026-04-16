#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from extractors import ChannelConfig
from mhc.constants import STATIONARY_WORKOUT_CHANNELS, WATCH_DISTANCE_CHANNEL, WATCH_HR_CHANNEL, WATCH_STEP_CHANNEL
from synthesizers import WorkoutSynthesizer
from synthesizers._helper import (
    append_distance_metrics,
    append_hr_metrics,
    append_step_metrics,
    index_or_none,
)
from timef.schema import Recording


class StationaryActivitySynthesizer(WorkoutSynthesizer):
    def __init__(
        self,
        min_duration: int = 0,
        hr_floor_bpm: float = 100.0,
        hr_margin_bpm: float = 15.0,
        max_distance_per_min: float = 3.0,
    ):
        super().__init__(min_duration=min_duration)
        self.hr_floor_bpm = hr_floor_bpm
        self.hr_margin_bpm = hr_margin_bpm
        self.max_distance_per_min = max_distance_per_min

    @property
    def template_key(self) -> str:
        return "stationary_activity"

    @property
    def workout_channels(self) -> list[tuple[str, str, str]]:
        return STATIONARY_WORKOUT_CHANNELS

    def _metrics_suffix(
        self,
        config: ChannelConfig,
        _label: str,
        row: Recording,
        start: int,
        end: int,
        channel_idxs: list[int],
    ) -> str:
        hr_idx = index_or_none(row, WATCH_HR_CHANNEL)
        distance_idx = index_or_none(row, WATCH_DISTANCE_CHANNEL)
        step_idx = index_or_none(row, WATCH_STEP_CHANNEL)
        parts: list[str] = []

        append_hr_metrics(
            parts,
            channel_idxs,
            config,
            row,
            start,
            end,
            hr_idx,
            elevated_threshold=self.hr_floor_bpm,
            include_space_before_day_unit=False,
        )
        append_distance_metrics(parts, channel_idxs, row, start, end, distance_idx)
        append_step_metrics(parts, channel_idxs, row, start, end, step_idx)

        if not parts:
            return ""
        return " " + " ".join(parts)
