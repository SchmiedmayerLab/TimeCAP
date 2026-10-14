#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import abc
import json

import numpy as np

from extractors import ChannelConfig
from synthesizers._base import CrossChannelSynthesizer
from synthesizers._helper import contiguous_windows, finalize_caption_text
from timef.schema import Annotation, Recording
from util import seed_from_key


class WorkoutSynthesizer(CrossChannelSynthesizer, abc.ABC):
    def __init__(self, min_duration: int = 0):
        self.min_duration = min_duration

    @property
    @abc.abstractmethod
    def template_key(self) -> str:
        """Template key for cross-channel workout captions."""

    @property
    @abc.abstractmethod
    def workout_channels(self) -> list[tuple[str, str, str]]:
        """Triples of workout channel, annotation label, and activity name."""

    @abc.abstractmethod
    def _metrics_suffix(
        self,
        config: ChannelConfig,
        label: str,
        row: Recording,
        start: int,
        end: int,
        channel_idxs: list[int],
    ) -> str:
        """Build the optional metrics suffix and append referenced channel indices."""

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        templates = json.loads(config.templates_path.read_text())["cross_channel"][self.template_key]
        time_unit = "hour" if config.time_unit == "hours" else "minute"
        seed = seed_from_key(row.row_id)

        results: list[Annotation] = []
        for workout_channel, label, activity_name in self.workout_channels:
            try:
                workout_idx = row.channel_names.index(workout_channel)
            except ValueError:
                continue

            workout = np.asarray(row.values[workout_idx], dtype=float)
            workout_active = np.isfinite(workout) & (workout > 0)
            if not workout_active.any():
                continue

            windows = contiguous_windows(workout_active, self.min_duration)
            for i, (start, end) in enumerate(windows):
                template = templates[(seed + i) % len(templates)]
                channel_idxs = [workout_idx]
                metrics_suffix = self._metrics_suffix(
                    config=config,
                    label=label,
                    row=row,
                    start=start,
                    end=end,
                    channel_idxs=channel_idxs,
                )
                text = template.format(
                    activity_name=activity_name,
                    time_unit=time_unit,
                    start=start,
                    end=end,
                    metrics_suffix=metrics_suffix,
                )
                text = finalize_caption_text(text, metrics_suffix)
                results.append(
                    Annotation(
                        caption_type="cross_channel",
                        text=text,
                        channel_idxs=tuple(channel_idxs),
                        window=(start, end),
                        label=label,
                    )
                )
        return results
