#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np


class MetricAggregator:
    def prepare(self, series: np.ndarray) -> np.ndarray:
        return series  # identity by default

    def aggregate(self, series: np.ndarray) -> tuple | None:
        prepared = self.prepare(series)
        if len(prepared) == 0:
            return None
        return (float(np.mean(prepared)), float(np.max(prepared)),
                float(np.min(prepared)), float(np.std(prepared)))


class NonZeroAggregator(MetricAggregator):
    """Filters out zero values before computing stats (e.g. for heart rate)."""
    def prepare(self, series: np.ndarray) -> np.ndarray:
        return series[series != 0]
