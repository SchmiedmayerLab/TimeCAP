#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json

import numpy as np

from extractors import ChannelConfig
from mhc.constants import STATIONARY_WORKOUT_CHANNELS, WATCH_DISTANCE_CHANNEL, WATCH_HR_CHANNEL, WATCH_STEP_CHANNEL
from synthesizers import CrossChannelSynthesizer, contiguous_windows
from timef.schema import Annotation, Recording
from util import seed_from_key


class StationaryActivitySynthesizer(CrossChannelSynthesizer):
    def __init__(
        self,
        min_duration: int = 0,
        hr_floor_bpm: float = 100.0,
        hr_margin_bpm: float = 15.0,
        max_distance_per_min: float = 3.0,
    ):
        self.min_duration = min_duration
        self.hr_floor_bpm = hr_floor_bpm
        self.hr_margin_bpm = hr_margin_bpm
        self.max_distance_per_min = max_distance_per_min

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        hr_idx = self._index_or_none(row, WATCH_HR_CHANNEL)
        distance_idx = self._index_or_none(row, WATCH_DISTANCE_CHANNEL)
        step_idx = self._index_or_none(row, WATCH_STEP_CHANNEL)

        templates = json.loads(config.templates_path.read_text())["cross_channel"]["stationary_activity"]
        time_unit = "hour" if config.time_unit == "hours" else "minute"
        seed = seed_from_key(row.row_id)

        results: list[Annotation] = []
        for workout_channel, label, activity_name in STATIONARY_WORKOUT_CHANNELS:
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
                end_inclusive = max(start, end - 1)
                template = templates[(seed + i) % len(templates)]
                channel_idxs = [workout_idx]
                metrics_suffix = self._metrics_suffix(
                    config=config,
                    row=row,
                    start=start,
                    end=end,
                    hr_idx=hr_idx,
                    distance_idx=distance_idx,
                    step_idx=step_idx,
                    channel_idxs=channel_idxs,
                )
                text = template.format(
                    activity_name=activity_name,
                    time_unit=time_unit,
                    start=start,
                    end=end_inclusive,
                    metrics_suffix=metrics_suffix,
                )
                text = self._finalize_caption_text(text, metrics_suffix)
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

    def _metrics_suffix(
        self,
        config: ChannelConfig,
        row: Recording,
        start: int,
        end: int,
        hr_idx: int | None,
        distance_idx: int | None,
        step_idx: int | None,
        channel_idxs: list[int],
    ) -> str:
        parts: list[str] = []

        self._append_hr_metrics(parts, channel_idxs, config, row, start, end, hr_idx)
        self._append_distance_metrics(parts, channel_idxs, row, start, end, distance_idx)

        step_mean = self._metric_mean(row, step_idx, start, end)
        step_total = self._metric_total(row, step_idx, start, end)
        if step_mean is not None and step_idx is not None:
            parts.append(
                self._sentence(f"The watch recorded an average step count of {step_mean:.1f} steps/min during this period")
            )
            if step_total is not None:
                parts.append(self._sentence(f"The total step count recorded by the watch during that time was {step_total:.0f}"))
            channel_idxs.append(step_idx)

        if not parts:
            return ""
        return " " + " ".join(parts)

    def _append_hr_metrics(
        self,
        parts: list[str],
        channel_idxs: list[int],
        config: ChannelConfig,
        row: Recording,
        start: int,
        end: int,
        hr_idx: int | None,
    ) -> None:
        hr_mean = self._metric_mean(row, hr_idx, start, end)
        if hr_mean is None or hr_idx is None:
            return

        hr_peak = self._metric_peak(row, hr_idx, start, end)
        parts.append(
            self._sentence(
                self._format_metric_summary(
                    config=config,
                    channel_name=WATCH_HR_CHANNEL,
                    mean=hr_mean,
                    peak=hr_peak,
                    elevated_threshold=self.hr_floor_bpm,
                )
            )
        )

        hr_day_delta = self._metric_day_mean_delta(row, hr_idx, start, end)
        hr_day_mean = self._metric_day_mean(row, hr_idx)
        if hr_day_delta is not None and hr_day_mean is not None:
            hr_name, hr_unit = self._channel_meta(config, WATCH_HR_CHANNEL)
            direction = "higher" if hr_day_delta >= 0 else "lower"
            parts.append(
                self._sentence(
                    f"The {hr_name} was {abs(hr_day_delta):.0f} {hr_unit} {direction} than the day's mean of {hr_day_mean:.0f}{hr_unit}"
                )
            )

        channel_idxs.append(hr_idx)

    def _append_distance_metrics(
        self,
        parts: list[str],
        channel_idxs: list[int],
        row: Recording,
        start: int,
        end: int,
        distance_idx: int | None,
    ) -> None:
        distance_mean = self._metric_mean(row, distance_idx, start, end)
        distance_total = self._metric_total(row, distance_idx, start, end)
        if distance_mean is None or distance_idx is None:
            return

        parts.append(self._sentence(f"The watch recorded an average distance of {distance_mean:.1f} m/min during this period"))
        if distance_total is not None:
            parts.append(self._sentence(f"The total distance recorded by the watch in that interval was {distance_total:.1f} m"))

        channel_idxs.append(distance_idx)
