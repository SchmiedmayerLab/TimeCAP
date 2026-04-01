#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from typing import Any, NamedTuple


class SignalData(NamedTuple):
    id: str
    spec_id: int
    name: str
    data: Any           # np.ndarray — 1-D time series for the channel
    metadata: dict


class CaptionData(NamedTuple):
    signal_id: str
    sample_id: str      # pre-computed, e.g. "mhc:u:d:heart_rate:statistical"
    spec_id: int
    answer: str
    window: tuple[int, int] | None  # None = full signal (statistical); (start, end) for structural
