#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import abc

from extractors import ChannelConfig
from timef.schema import Annotation, Recording


class CrossChannelSynthesizer(abc.ABC):
    @abc.abstractmethod
    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]: ...
