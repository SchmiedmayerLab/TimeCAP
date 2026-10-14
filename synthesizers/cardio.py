#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from extractors import ChannelConfig
from mhc.constants import (
    CARDIO_WORKOUT_CHANNELS,
    ELEVATED_HEART_RATE_BPM,
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
)
from timef.schema import Recording


class CardioSynthesizer(WorkoutSynthesizer):
    def __init__(
        self,
        min_duration: int = 0,
        hr_elevated_threshold_bpm: float = ELEVATED_HEART_RATE_BPM,
    ):
        super().__init__(min_duration=min_duration)
        self.hr_elevated_threshold_bpm = hr_elevated_threshold_bpm

    @property
    def template_key(self) -> str:
        return "cardio"

    @property
    def workout_channels(self) -> list[tuple[str, str, str]]:
        return CARDIO_WORKOUT_CHANNELS

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

        append_hr_metrics(parts, channel_idxs, config, row, start, end, hr_idx, elevated_threshold=self.hr_elevated_threshold_bpm)
        append_distance_metrics(parts, channel_idxs, config, row, start, end, distance_idx)
        append_active_energy_metrics(parts, channel_idxs, config, row, start, end, energy_idx)

        if label == "cardio_running":
            append_step_metrics(parts, channel_idxs, config, row, start, end, step_idx)

        if not parts:
            return ""
        return " " + " ".join(parts)


class EnduranceActivitySynthesizer(WorkoutSynthesizer):
    @property
    def template_key(self) -> str:
        return "endurance_activity"

    @property
    def workout_channels(self) -> list[tuple[str, str, str]]:
        return [
            (
                "workout:HKWorkoutActivityTypeElliptical",
                "endurance_elliptical",
                "an elliptical workout",
            ),
            (
                "workout:HKWorkoutActivityTypeMixedMetabolicCardioTraining",
                "endurance_mixed_cardio",
                "a mixed cardio workout",
            ),
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
        energy_idx = index_or_none(row, WATCH_ACTIVE_ENERGY_CHANNEL)
        parts: list[str] = []

        append_hr_metrics(parts, channel_idxs, config, row, start, end, hr_idx, elevated_threshold=100.0)
        append_distance_metrics(parts, channel_idxs, config, row, start, end, distance_idx)
        append_active_energy_metrics(parts, channel_idxs, config, row, start, end, energy_idx)

        if not parts:
            return ""
        return " " + " ".join(parts)
