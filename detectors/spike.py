#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np

from detectors import DetectionResult, StructuralDetector


class SpikeDetector(StructuralDetector):
    """Detects spikes and drops.
    """

    def __init__(self, filter_zeros: bool = False) -> None:
        super().__init__(filter_zeros=filter_zeros)

    def _detect(self, series: np.ndarray, indices: np.ndarray) -> list[DetectionResult]:
        return []
