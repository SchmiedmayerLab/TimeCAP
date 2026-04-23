#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from extractors import CaptionExtractor, ChannelConfig
from synthesizers import CrossChannelSynthesizer
from synthesizers.cardio import CardioSynthesizer
from synthesizers.sleep import SleepSynthesizer
from synthesizers.stationary_activity import StationaryActivitySynthesizer
from timef.schema import Annotation, Recording


class CrossChannelExtractor(CaptionExtractor):
    caption_type = "cross_channel"

    def __init__(self, config: ChannelConfig, synthesizers: list[CrossChannelSynthesizer]):
        super().__init__(config)
        self.synthesizers = synthesizers

    def extract(self, row: Recording) -> list[Annotation]:
        results: list[Annotation] = []
        for synth in self.synthesizers:
            results.extend(synth.synthesize(row, self.config))
        return results


def _default_min_duration(config: ChannelConfig) -> int:
    return 1 if config.time_unit == "hours" else 5


def build_default_cross_channel_synthesizers(config: ChannelConfig) -> list[CrossChannelSynthesizer]:
    min_duration = _default_min_duration(config)
    return [
        SleepSynthesizer(min_duration=min_duration),
        StationaryActivitySynthesizer(min_duration=min_duration),
        CardioSynthesizer(min_duration=min_duration),
    ]


def build_default_cross_channel_extractor(config: ChannelConfig) -> CrossChannelExtractor:
    return CrossChannelExtractor(config, synthesizers=build_default_cross_channel_synthesizers(config))
