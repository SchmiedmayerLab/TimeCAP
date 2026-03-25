#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import pathlib

import numpy as np

from aggregators import MetricAggregator
from extractors import CaptionExtractor
from timef.schema import Annotation, Sample, SampleRef, SampleSignalRef, Signal

TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent.parent / "templates" / "templates.json"
STAT_TEMPLATES: list[str] = json.loads(TEMPLATES_PATH.read_text())["statistical"]

DEFAULT_AGGREGATOR = MetricAggregator()


class StatisticalExtractor(CaptionExtractor):
    caption_type = "statistical"

    def extract(self, signals: list[Signal]) -> list[tuple[Sample, Annotation]]:
        if not signals:
            raise ValueError("signals must not be empty")
        seed = self._seed(signals[0].id)
        results = []

        for i, signal in enumerate(signals):
            if signal.name not in self.config.continuous:
                continue

            # remove data gaps
            series = signal.data[~np.isnan(signal.data)].astype(float)

            aggregator = self.config.aggregators.get(signal.name, DEFAULT_AGGREGATOR)
            stats = aggregator.aggregate(series)
            if stats is None:
                continue
            mean_v, max_v, min_v, std_v = stats

            name, unit, decimals = self.config.meta[signal.name]
            format = lambda v, d=decimals: f"{v:.{d}f}"

            template = STAT_TEMPLATES[(seed + i) % len(STAT_TEMPLATES)]
            caption = template.format(
                name=name, unit=unit,
                mean=format(mean_v), max=format(max_v),
                min=format(min_v),   std=format(std_v),
            )

            sample_id = f"{signal.id}:{self.caption_type}"
            sample = Sample(id=sample_id, signals=[SampleSignalRef(signal_id=signal.id)])
            annotation = Annotation(
                id=f"{signal.id}:{self.caption_type}",
                spec_id=f"captioning:{self.caption_type}",
                samples=[SampleRef(sample_id=sample_id)],
                answer=caption,
            )
            results.append((sample, annotation))

        return results
