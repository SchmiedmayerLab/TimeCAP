#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Minimal helpers for clinical synthesizers.

Copies of the pure-numpy utilities from synthesizers/_helper.py, isolated so
the clinical module does not trigger the mhc package-level import chain
(mhc/__init__.py → mhc.dataset → torch → heavy dependency cascade).
"""
from __future__ import annotations

import numpy as np

from clinical.constants import SLEEP_ASLEEP_CHANNEL, WATCH_HR_CHANNEL
from timef.schema import Recording


def contiguous_windows(mask: np.ndarray, min_duration: int) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    keep = (ends - starts) >= min_duration
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))


def index_or_none(row: Recording, channel_name: str) -> int | None:
    try:
        return row.channel_names.index(channel_name)
    except ValueError:
        return None


def sleep_hr_windows(
    row: Recording, min_duration: int
) -> tuple[int, int, list[tuple[int, int, np.ndarray, np.ndarray]]] | None:
    """Sleep windows of at least ``min_duration`` minutes that contain valid HR.

    Returns (sleep_idx, hr_idx, windows) with each window as (start, end, hr_window, valid),
    or None when the sleep or HR channel is missing.
    """
    sleep_idx = index_or_none(row, SLEEP_ASLEEP_CHANNEL)
    hr_idx = index_or_none(row, WATCH_HR_CHANNEL)
    if sleep_idx is None or hr_idx is None:
        return None
    sleep = np.asarray(row.values[sleep_idx], dtype=float)
    hr = np.asarray(row.values[hr_idx], dtype=float)
    windows = []
    for start, end in contiguous_windows(np.isfinite(sleep) & (sleep > 0), min_duration):
        hr_window = hr[start:end]
        valid = np.isfinite(hr_window) & (hr_window > 0)
        if valid.any():
            windows.append((start, end, hr_window, valid))
    return sleep_idx, hr_idx, windows
