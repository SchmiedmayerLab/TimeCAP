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

    @classmethod
    def default_synthesizers(cls, config: ChannelConfig) -> list[CrossChannelSynthesizer]:
        min_duration = 1 if config.time_unit == "hours" else 5
        return [
            SleepSynthesizer(min_duration=min_duration),
            StationaryActivitySynthesizer(min_duration=min_duration),
            CardioSynthesizer(min_duration=min_duration),
        ]
