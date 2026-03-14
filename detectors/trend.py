from __future__ import annotations

from typing import List

import numpy as np

from detectors import DetectionResult, StructuralDetector


class TrendDetector(StructuralDetector):
    """Detects statistically significant trends.
    """

    def __init__(self, filter_zeros: bool = False) -> None:
        super().__init__(filter_zeros=filter_zeros)

    def _detect(self, series: np.ndarray, indices: np.ndarray) -> List[DetectionResult]:
        return []
