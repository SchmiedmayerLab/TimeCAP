#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AggregationResult:
    stats: tuple[float, float, float, float] | None = None
    total: float | None = None
    duration_units: float | None = None


class MetricAggregator:
    def prepare(self, series: np.ndarray) -> np.ndarray:
        return series  # identity by default

    def aggregate(self, series: np.ndarray) -> AggregationResult | None:
        prepared = self.prepare(series)
        if len(prepared) == 0:
            return None
        return AggregationResult(
            stats=(
                float(np.mean(prepared)),
                float(np.max(prepared)),
                float(np.min(prepared)),
                float(np.std(prepared)),
            )
        )


class NonZeroAggregator(MetricAggregator):
    """Filters out zero values before computing stats (e.g. for heart rate)."""
    def prepare(self, series: np.ndarray) -> np.ndarray:
        return series[series != 0]


class TotalAggregator(MetricAggregator):
    def aggregate(self, series: np.ndarray) -> AggregationResult | None:
        prepared = self.prepare(series)
        if len(prepared) == 0:
            return None
        return AggregationResult(
            stats=(
                float(np.mean(prepared)),
                float(np.max(prepared)),
                float(np.min(prepared)),
                float(np.std(prepared)),
            ),
            total=float(np.sum(prepared)),
        )


class DurationAggregator(MetricAggregator):
    """Sums positive activity/sleep bins into a total duration."""

    def prepare(self, series: np.ndarray) -> np.ndarray:
        return series[series > 0]

    def aggregate(self, series: np.ndarray) -> AggregationResult | None:
        prepared = self.prepare(series)
        if len(prepared) == 0:
            return None
        return AggregationResult(duration_units=float(np.sum(prepared)))
