#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from extractors import ChannelConfig
from mhc.constants import WATCH_ACTIVE_ENERGY_CHANNEL, WATCH_DISTANCE_CHANNEL, WATCH_HR_CHANNEL
from synthesizers import WorkoutSynthesizer
from synthesizers._helper import (
    append_active_energy_metrics,
    append_distance_metrics,
    append_hr_metrics,
    index_or_none,
)
from timef.schema import Recording


class OtherActivitySynthesizer(WorkoutSynthesizer):
    @property
    def template_key(self) -> str:
        return "other_activity"

    @property
    def workout_channels(self) -> list[tuple[str, str, str]]:
        return [
            (
                "workout:HKWorkoutActivityTypeOther",
                "other_activity",
                "another workout",
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
        energy_idx = index_or_none(row, WATCH_ACTIVE_ENERGY_CHANNEL)
        parts: list[str] = []

        append_hr_metrics(parts, channel_idxs, config, row, start, end, hr_idx, elevated_threshold=100.0)
        append_distance_metrics(parts, channel_idxs, config, row, start, end, distance_idx)
        append_active_energy_metrics(parts, channel_idxs, config, row, start, end, energy_idx)

        if not parts:
            return ""
        return " " + " ".join(parts)
