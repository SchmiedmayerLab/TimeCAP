#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json

import numpy as np

from extractors import ChannelConfig
from mhc.constants import WATCH_DISTANCE_CHANNEL, WATCH_HR_CHANNEL, WATCH_STEP_CHANNEL
from synthesizers import CrossChannelSynthesizer
from synthesizers._helper import append_hr_metrics, contiguous_windows, index_or_none, render_metric_text, sentence
from timef.schema import Annotation, Recording
from util import capitalize, seed_from_key


class SleepSynthesizer(CrossChannelSynthesizer):
    def __init__(self, min_duration: int = 0):
        self.min_duration = min_duration

    @property
    def sleep_channels(self) -> list[tuple[str, str, str]]:
        return [
            ("sleep:asleep", "sleep_asleep", "asleep"),
            ("sleep:inbed", "sleep_inbed", "in bed"),
        ]

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        time_unit = "hour" if config.time_unit == "hours" else "minute"
        cross_channel = json.loads(config.templates_path.read_text())["cross_channel"]
        seed = seed_from_key(row.row_id)

        results = self._synthesize_sleep_stages(row, config, cross_channel, time_unit, seed)
        results.extend(self._synthesize_in_bed_not_sleeping(row, cross_channel, time_unit, seed))
        return results

    def _synthesize_sleep_stages(
        self,
        row: Recording,
        config: ChannelConfig,
        cross_channel: dict,
        time_unit: str,
        seed: int,
    ) -> list[Annotation]:
        templates = cross_channel["sleep_stage"]
        results: list[Annotation] = []

        for channel_name, label, sleep_state in self.sleep_channels:
            try:
                sleep_idx = row.channel_names.index(channel_name)
            except ValueError:
                continue

            sleep = np.asarray(row.values[sleep_idx], dtype=float)
            sleep_active = np.isfinite(sleep) & (sleep > 0)
            if not sleep_active.any():
                continue

            for i, (start, end) in enumerate(contiguous_windows(sleep_active, self.min_duration)):
                template = templates[(seed + i) % len(templates)]
                channel_idxs = [sleep_idx]
                metrics_suffix = self._metrics_suffix(
                    config=config,
                    row=row,
                    start=start,
                    end=end,
                    channel_idxs=channel_idxs,
                )
                text = template.format(
                    sleep_state=sleep_state,
                    time_unit=time_unit,
                    start=start,
                    end=end,
                    metrics_suffix=metrics_suffix,
                )
                results.append(
                    Annotation(
                        caption_type="cross_channel",
                        text=capitalize(text.rstrip(".") + "."),
                        channel_idxs=tuple(dict.fromkeys(channel_idxs)),
                        window=(start, end),
                        label=label,
                    )
                )

        return results

    def _synthesize_in_bed_not_sleeping(
        self,
        row: Recording,
        cross_channel: dict,
        time_unit: str,
        seed: int,
    ) -> list[Annotation]:
        try:
            in_bed_idx = row.channel_names.index("sleep:inbed")
            asleep_idx = row.channel_names.index("sleep:asleep")
        except ValueError:
            return []

        in_bed = np.asarray(row.values[in_bed_idx], dtype=float)
        asleep = np.asarray(row.values[asleep_idx], dtype=float)
        if not np.any(~np.isnan(asleep)):
            return []
        # Mark intervals where the user is in bed but not recorded as asleep.
        mask = (~np.isnan(in_bed)) & (in_bed > 0) & ~((~np.isnan(asleep)) & (asleep > 0))

        templates = cross_channel["sleep"]

        results: list[Annotation] = []
        for i, (start, end) in enumerate(contiguous_windows(mask, self.min_duration)):
            template = templates[(seed + i) % len(templates)]
            text = template.format(time_unit=time_unit, start=start, end=end)
            results.append(
                Annotation(
                    caption_type="cross_channel",
                    text=capitalize(text),
                    channel_idxs=(asleep_idx, in_bed_idx),
                    window=(start, end),
                    label="in_bed_not_sleeping",
                )
            )
        return results

    @staticmethod
    def _metrics_suffix(
        config: ChannelConfig,
        row: Recording,
        start: int,
        end: int,
        channel_idxs: list[int],
    ) -> str:
        hr_idx = index_or_none(row, WATCH_HR_CHANNEL)
        distance_idx = index_or_none(row, WATCH_DISTANCE_CHANNEL)
        step_idx = index_or_none(row, WATCH_STEP_CHANNEL)
        parts: list[str] = []

        append_hr_metrics(parts, channel_idxs, config, row, start, end, hr_idx, elevated_threshold=100.0)
        _append_stationary_sleep_metric(parts, channel_idxs, config, row, start, end, distance_idx, step_idx)

        if not parts:
            return ""
        return " " + " ".join(parts)


def _append_stationary_sleep_metric(
    parts: list[str],
    channel_idxs: list[int],
    config: ChannelConfig,
    row: Recording,
    start: int,
    end: int,
    distance_idx: int | None,
    step_idx: int | None,
) -> None:
    if distance_idx is None:
        return

    distance_values = np.asarray(row.values[distance_idx][start:end], dtype=float)
    distance_finite = distance_values[np.isfinite(distance_values)]
    if distance_finite.size == 0:
        return

    distance_total = float(np.sum(np.clip(distance_finite, 0, None)))
    if distance_total > 0:
        return

    if step_idx is not None:
        step_values = np.asarray(row.values[step_idx][start:end], dtype=float)
        step_finite = step_values[np.isfinite(step_values)]
        if step_finite.size > 0:
            step_total = float(np.sum(np.clip(step_finite, 0, None)))
            if step_total > 0:
                return

    if distance_idx not in channel_idxs:
        channel_idxs.append(distance_idx)
    if step_idx is not None and step_idx not in channel_idxs:
        channel_idxs.append(step_idx)

    parts.append(
        sentence(
            render_metric_text(
                config,
                row,
                start,
                end,
                "sleep_stationary",
                "The person was stationary during this sleep interval",
            )
        )
    )
