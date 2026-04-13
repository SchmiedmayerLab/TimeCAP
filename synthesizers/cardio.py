#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json

import numpy as np

from extractors import CaptionExtractor, ChannelConfig
from mhc.constants import CARDIO_WORKOUT_CHANNELS, WATCH_DISTANCE_CHANNEL, WATCH_HR_CHANNEL, WATCH_STEP_CHANNEL
from synthesizers import CrossChannelSynthesizer, contiguous_windows
from timef.schema import Annotation, Recording


class CardioSynthesizer(CrossChannelSynthesizer):
    def __init__(self, min_duration: int = 0):
        self.min_duration = min_duration

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        hr_idx = self._index_or_none(row, WATCH_HR_CHANNEL)
        distance_idx = self._index_or_none(row, WATCH_DISTANCE_CHANNEL)
        step_idx = self._index_or_none(row, WATCH_STEP_CHANNEL)

        templates = json.loads(config.templates_path.read_text())["cross_channel"]["cardio"]
        time_unit = "hour" if config.time_unit == "hours" else "minute"
        seed = CaptionExtractor._seed(row.row_id)

        results: list[Annotation] = []
        for workout_channel, label, template_key in CARDIO_WORKOUT_CHANNELS:
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
                subtype_templates = templates[template_key]
                template = subtype_templates[(seed + i) % len(subtype_templates)]
                channel_idxs = [workout_idx]
                metrics_suffix = self._metrics_suffix(
                    template_key=template_key,
                    row=row,
                    start=start,
                    end=end,
                    hr_idx=hr_idx,
                    distance_idx=distance_idx,
                    step_idx=step_idx,
                    channel_idxs=channel_idxs,
                )
                text = template.format(
                    time_unit=time_unit,
                    start=start,
                    end=end_inclusive,
                    metrics_suffix=metrics_suffix,
                )
                results.append(
                    Annotation(
                        caption_type="cross_channel",
                        text=text,
                        channel_idxs=tuple(channel_idxs),
                        window=(start, end),
                        label=label,
                    )
                )
            template_offset += len(windows)
        return results

    def _metrics_suffix(
        self,
        template_key: str,
        row: Recording,
        start: int,
        end: int,
        hr_idx: int | None,
        distance_idx: int | None,
        step_idx: int | None,
        channel_idxs: list[int],
    ) -> str:
        parts: list[str] = []

        hr_mean = self._metric_mean(row, hr_idx, start, end)
        if hr_mean is not None and hr_idx is not None:
            hr_peak = self._metric_peak(row, hr_idx, start, end)
            if hr_peak is not None:
                parts.append(f"avg HR {hr_mean:.0f} bpm, peak HR {hr_peak:.0f} bpm")
            else:
                parts.append(f"avg HR {hr_mean:.0f} bpm")
            channel_idxs.append(hr_idx)

        distance_mean = self._metric_mean(row, distance_idx, start, end)
        distance_total = self._metric_total(row, distance_idx, start, end)
        if distance_mean is not None and distance_idx is not None:
            if template_key == "cycling" and distance_total is not None:
                parts.append(
                    f"avg watch distance {distance_mean:.1f} m/min, total watch distance {distance_total:.1f} m"
                )
            elif template_key == "running" and distance_total is not None:
                parts.append(
                    f"avg watch distance {distance_mean:.1f} m/min, total watch distance {distance_total:.1f} m"
                )
            else:
                parts.append(f"avg watch distance {distance_mean:.1f} m/min")
            channel_idxs.append(distance_idx)

        if template_key == "running":
            step_mean = self._metric_mean(row, step_idx, start, end)
            step_total = self._metric_total(row, step_idx, start, end)
            if step_mean is not None and step_idx is not None:
                if step_total is not None:
                    parts.append(f"avg watch steps {step_mean:.1f} steps/min, total watch steps {step_total:.0f}")
                else:
                    parts.append(f"avg watch steps {step_mean:.1f} steps/min")
                channel_idxs.append(step_idx)

        if not parts:
            return ""
        return ", " + ", ".join(parts)
