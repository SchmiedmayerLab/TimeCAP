#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json

import numpy as np

from aggregators import MetricAggregator
from extractors import CaptionExtractor, ChannelConfig
from timef.schema import Annotation, Recording
from util import capitalize

DEFAULT_AGGREGATOR = MetricAggregator()


class StatisticalExtractor(CaptionExtractor):
    caption_type = "statistical"

    def __init__(self, config: ChannelConfig):
        super().__init__(config)
        self.templates: list[str] = json.loads(config.templates_path.read_text())["statistical"]

    def extract(self, row: Recording) -> list[Annotation]:
        seed = self._seed(row.row_id)
        results: list[Annotation] = []

        for i, signal in enumerate(row.iter_channels()):
            if signal.name not in self.config.continuous:
                continue

            series = signal.data[~np.isnan(signal.data)].astype(float)

            aggregator = self.config.aggregators.get(signal.name, DEFAULT_AGGREGATOR)
            stats = aggregator.aggregate(series)
            if stats is None:
                continue
            mean_v, max_v, min_v, std_v = stats

            name, unit, decimals = self.config.meta[signal.name]
            format = lambda v, d=decimals: f"{v:.{d}f}"

            template = self.templates[(seed + i) % len(self.templates)]
            caption = capitalize(template.format(
                name=name, unit=unit,
                mean=format(mean_v), max=format(max_v),
                min=format(min_v),   std=format(std_v),
            ))
            results.append(Annotation(caption_type=self.caption_type, text=caption, channel_idxs=(i,)))

        return results
