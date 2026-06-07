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
