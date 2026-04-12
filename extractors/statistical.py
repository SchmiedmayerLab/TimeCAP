#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json

import numpy as np

from aggregators import AggregationResult, MetricAggregator
from extractors import CaptionExtractor, ChannelConfig
from timef.schema import Annotation, Recording

DEFAULT_AGGREGATOR = MetricAggregator()


class StatisticalExtractor(CaptionExtractor):
    caption_type = "statistical"

    def __init__(self, config: ChannelConfig):
        super().__init__(config)
        self.templates: list[str] = json.loads(config.templates_path.read_text())["statistical"]

    def extract(self, row: Recording) -> list[Annotation]:
        seed = self._seed(row.row_id)
        results: list[Annotation] = []
        statistical_names = self.config.continuous | frozenset(self.config.aggregators.keys())

        for i, signal in enumerate(row.iter_channels()):
            if signal.name not in statistical_names:
                continue

            series = signal.data[~np.isnan(signal.data)].astype(float)

            aggregator = self.config.aggregators.get(signal.name, DEFAULT_AGGREGATOR)
            aggregated = aggregator.aggregate(series)
            if aggregated is None:
                continue

            if signal.name in self.config.meta:
                name, unit, decimals = self.config.meta[signal.name]
            else:
                name, unit, decimals = self.config.display_name(signal.name), "", 0
            format = lambda v, d=decimals: f"{v:.{d}f}"

            if aggregated.stats is not None:
                mean_v, max_v, min_v, std_v = aggregated.stats
                template = self.templates[(seed + i) % len(self.templates)]
                caption = template.format(
                    name=name, unit=unit,
                    mean=format(mean_v), max=format(max_v),
                    min=format(min_v),   std=format(std_v),
                )
                results.append(Annotation(caption_type=self.caption_type, text=caption, channel_idxs=(i,)))

            extra_captions = self._extra_captions(signal.name, name, unit, aggregated, format, row)
            results.extend(
                Annotation(caption_type=self.caption_type, text=caption, channel_idxs=(i,))
                for caption in extra_captions
            )

        return results

    def _extra_captions(
        self,
        channel_name: str,
        display_name: str,
        unit: str,
        aggregated: AggregationResult,
        format_value,
        row: Recording,
    ) -> list[str]:
        captions: list[str] = []

        if aggregated.total is not None:
            period_label = self._period_label(row)
            total_unit = self._total_unit(unit)
            total_value = format_value(aggregated.total)
            suffix = f" {total_unit}" if total_unit else ""
            captions.append(f"{period_label} total {display_name}: {total_value}{suffix}.")

        if aggregated.duration_units is not None:
            duration = self._format_duration(aggregated.duration_units)
            activity_name = self._duration_label(channel_name)
            captions.append(f"{duration} {activity_name}.")

        return captions

    def _period_label(self, row: Recording) -> str:
        if self.config.time_unit == "minutes" and row.values.shape[1] == 1440:
            return "Daily"
        if self.config.time_unit == "hours" and row.values.shape[1] == 24 * 7:
            return "Weekly"
        return "Total"

    @staticmethod
    def _total_unit(unit: str) -> str:
        if not unit:
            return ""
        base = unit.split("/", 1)[0]
        return "" if base == "count" else base

    def _format_duration(self, duration_units: float) -> str:
        if self.config.time_unit == "minutes":
            total_minutes = int(round(duration_units))
            hours, minutes = divmod(total_minutes, 60)
            if hours and minutes:
                return f"{hours}h {minutes}m"
            if hours:
                return f"{hours}h"
            return f"{minutes}m"
        if self.config.time_unit == "hours":
            rounded = int(round(duration_units))
            return f"{rounded}h"
        return f"{duration_units:g} {self.config.time_unit}"

    def _duration_label(self, channel_name: str) -> str:
        if channel_name == "sleep:inbed":
            return "in bed"
        if channel_name == "sleep:asleep":
            return "sleeping"
        return self.config.display_name(channel_name)
