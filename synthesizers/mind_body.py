#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from extractors import ChannelConfig
from mhc.constants import (
    WATCH_ACTIVE_ENERGY_CHANNEL,
    WATCH_DISTANCE_CHANNEL,
    WATCH_HR_CHANNEL,
    WATCH_STEP_CHANNEL,
)
from synthesizers import WorkoutSynthesizer
from synthesizers._helper import (
    append_active_energy_metrics,
    append_distance_metrics,
    append_hr_metrics,
    append_step_metrics,
    index_or_none,
    metric_mean,
)
from timef.schema import Recording


class MindBodyActivitySynthesizer(WorkoutSynthesizer):
    def __init__(
        self,
        min_duration: int = 0,
        hr_elevated_threshold_bpm: float = 100.0,
        min_movement_mean: float = 1.0,
    ):
        super().__init__(min_duration=min_duration)
        self.hr_elevated_threshold_bpm = hr_elevated_threshold_bpm
        self.min_movement_mean = min_movement_mean

    @property
    def template_key(self) -> str:
        return "mind_body_activity"

    @property
    def workout_channels(self) -> list[tuple[str, str, str]]:
        return [
            (
                "workout:HKWorkoutActivityTypeYoga",
                "mind_body_yoga",
                "yoga",
            )
        ]

    def _metrics_suffix(
        self,
        config: ChannelConfig,
        label: str,
        row: Recording,
        start: int,
        end: int,
        channel_idxs: list[int],
    ) -> str:
        hr_idx = index_or_none(row, WATCH_HR_CHANNEL)
        distance_idx = index_or_none(row, WATCH_DISTANCE_CHANNEL)
        step_idx = index_or_none(row, WATCH_STEP_CHANNEL)
        energy_idx = index_or_none(row, WATCH_ACTIVE_ENERGY_CHANNEL)
        parts: list[str] = []

        append_hr_metrics(
            parts,
            channel_idxs,
            config,
            row,
            start,
            end,
            hr_idx,
            elevated_threshold=self.hr_elevated_threshold_bpm,
        )
        append_active_energy_metrics(parts, channel_idxs, config, row, start, end, energy_idx)

        distance_mean = metric_mean(row, distance_idx, start, end)
        if distance_mean is not None and distance_mean >= self.min_movement_mean:
            append_distance_metrics(parts, channel_idxs, config, row, start, end, distance_idx)

        step_mean = metric_mean(row, step_idx, start, end)
        if step_mean is not None and step_mean >= self.min_movement_mean:
            append_step_metrics(parts, channel_idxs, config, row, start, end, step_idx)

        if not parts:
            return ""
        return " " + " ".join(parts)
