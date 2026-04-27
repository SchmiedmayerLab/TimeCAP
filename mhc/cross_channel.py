#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from extractors import ChannelConfig
from extractors.cross_channel import CrossChannelExtractor
from synthesizers import CrossChannelSynthesizer
from synthesizers.cardio import CardioSynthesizer
from synthesizers.sleep import SleepSynthesizer
from synthesizers.stationary_activity import StationaryActivitySynthesizer


def default_synthesizers(config: ChannelConfig) -> list[CrossChannelSynthesizer]:
    min_duration = 1 if config.time_unit == "hours" else 5
    return [
        SleepSynthesizer(min_duration=min_duration),
        StationaryActivitySynthesizer(min_duration=min_duration),
        CardioSynthesizer(min_duration=min_duration),
    ]


def default_extractor(config: ChannelConfig) -> CrossChannelExtractor:
    return CrossChannelExtractor(config, synthesizers=default_synthesizers(config))
