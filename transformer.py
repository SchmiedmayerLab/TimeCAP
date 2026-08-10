#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import abc

from timef.schema import Recording


class Transformer(abc.ABC):
    @abc.abstractmethod
    def transform_row(self, row: dict) -> Recording: ...
