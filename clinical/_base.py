#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Standalone base class for clinical synthesizers.

Duplicates synthesizers._base.CrossChannelSynthesizer to avoid importing the
synthesizers package, whose __init__ eagerly loads WorkoutSynthesizer →
synthesizers._helper → mhc.constants → mhc.__init__ → mhc.dataset → torch.
"""
from __future__ import annotations

import abc

from extractors import ChannelConfig
from timef.schema import Annotation, Recording


class ClinicalSynthesizer(abc.ABC):
    """Base class for clinical signal synthesizers.

    Identical contract to CrossChannelSynthesizer — instances can be passed
    to CrossChannelExtractor once the import chain is resolved in production.
    """
    @abc.abstractmethod
    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]: ...
