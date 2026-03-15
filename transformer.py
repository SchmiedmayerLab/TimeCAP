#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import abc

from timef.schema import SignalSpec, Signal


class Transformer(abc.ABC):
    @abc.abstractmethod
    def get_signal_specs(self) -> list[SignalSpec]: ...

    @abc.abstractmethod
    def transform_row(self, row: dict) -> list[Signal]: ...
