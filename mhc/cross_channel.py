#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from extractors import ChannelConfig
from extractors.cross_channel import CrossChannelExtractor
from synthesizers import CrossChannelSynthesizer
from synthesizers.cardio import CardioSynthesizer, EnduranceActivitySynthesizer
from synthesizers.locomotion import WalkingSynthesizer
from synthesizers.mind_body import MindBodyActivitySynthesizer
from synthesizers.other_activity import OtherActivitySynthesizer
from synthesizers.sleep import SleepSynthesizer
from synthesizers.stationary_activity import StationaryActivitySynthesizer


def default_synthesizers(config: ChannelConfig) -> list[CrossChannelSynthesizer]:
    min_duration = 1 if config.time_unit == "hours" else 5
    return [
        SleepSynthesizer(min_duration=min_duration),
        WalkingSynthesizer(min_duration=min_duration),
        EnduranceActivitySynthesizer(min_duration=min_duration),
        MindBodyActivitySynthesizer(min_duration=min_duration),
        OtherActivitySynthesizer(min_duration=min_duration),
        StationaryActivitySynthesizer(min_duration=min_duration),
        CardioSynthesizer(min_duration=min_duration),
    ]


def default_extractor(config: ChannelConfig) -> CrossChannelExtractor:
    return CrossChannelExtractor(config, synthesizers=default_synthesizers(config))
