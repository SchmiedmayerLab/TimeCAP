#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np

from detectors.trend import MonotonicTrendDetector


def test_detects_non_strict_decreasing_region() -> None:
    detector = MonotonicTrendDetector(min_points=3, min_span=2, min_length=1)

    results = detector.detect(np.array([5.0, 4.0, 4.0, 3.0]))

    assert len(results) == 1
    assert results[0].direction == "decreasing"
    assert results[0].window == (0, 3)


def test_merges_decreasing_regions_across_allowed_outlier() -> None:
    detector = MonotonicTrendDetector(min_points=2, min_span=1, min_length=1, max_outliers=1)

    results = detector.detect(np.array([10.0, 9.0, 8.0, 9.0, 7.0, 6.0]))

    assert len(results) == 1
    assert results[0].direction == "decreasing"
    assert results[0].window == (0, 5)


def test_does_not_merge_decreasing_regions_when_outlier_exceeds_start_value() -> None:
    detector = MonotonicTrendDetector(min_points=2, min_span=1, min_length=1, max_outliers=1)

    results = detector.detect(np.array([10.0, 9.0, 8.0, 12.0, 7.0, 6.0]))

    assert [(result.direction, result.window) for result in results] == [
        ("decreasing", (0, 2)),
        ("decreasing", (3, 5)),
    ]


def test_merges_increasing_regions_across_allowed_outlier() -> None:
    detector = MonotonicTrendDetector(min_points=2, min_span=1, min_length=1, max_outliers=1)

    results = detector.detect(np.array([1.0, 2.0, 3.0, 2.0, 4.0, 5.0]))

    assert len(results) == 1
    assert results[0].direction == "increasing"
    assert results[0].window == (0, 5)


def test_increasing_region_keeps_only_one_leading_zero() -> None:
    detector = MonotonicTrendDetector(min_points=3, min_span=2, min_length=1)

    results = detector.detect(np.array([0.0, 0.0, 0.0, 2.0, 3.0]))

    assert len(results) == 1
    assert results[0].direction == "increasing"
    assert results[0].window == (2, 4)


def test_decreasing_region_keeps_only_one_trailing_zero() -> None:
    detector = MonotonicTrendDetector(min_points=3, min_span=2, min_length=1)

    results = detector.detect(np.array([5.0, 4.0, 0.0, 0.0, 0.0]))

    assert len(results) == 1
    assert results[0].direction == "decreasing"
    assert results[0].window == (0, 2)


def test_increasing_region_keeps_only_one_leading_plateau_value() -> None:
    detector = MonotonicTrendDetector(min_points=3, min_span=2, min_length=1)

    results = detector.detect(np.array([1.0, 1.0, 1.0, 2.0, 3.0]))

    assert len(results) == 1
    assert results[0].direction == "increasing"
    assert results[0].window == (2, 4)


def test_increasing_region_keeps_only_one_trailing_plateau_value() -> None:
    detector = MonotonicTrendDetector(min_points=3, min_span=2, min_length=1)

    results = detector.detect(np.array([1.0, 2.0, 3.0, 3.0, 3.0]))

    assert len(results) == 1
    assert results[0].direction == "increasing"
    assert results[0].window == (0, 2)


def test_flat_region_is_not_a_trend() -> None:
    detector = MonotonicTrendDetector(min_points=3, min_span=2, min_length=1)

    results = detector.detect(np.array([2.0, 2.0, 2.0, 2.0]))

    assert results == []


def test_default_min_length_rejects_short_trend_after_merging() -> None:
    detector = MonotonicTrendDetector(min_points=2, min_span=1, max_outliers=1)

    results = detector.detect(np.array([1.0, 2.0, 3.0, 2.0, 4.0, 5.0]))

    assert results == []


def test_default_min_length_accepts_merged_trend_of_ten() -> None:
    detector = MonotonicTrendDetector(min_points=2, min_span=1, max_outliers=1)

    results = detector.detect(np.array([1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))

    assert len(results) == 1
    assert results[0].direction == "increasing"
    assert results[0].window == (0, 9)
