#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from extractors import CaptionExtractor, ChannelConfig
from models.base import BaseModel
from timef.schema import Annotation, Sample, Signal


class GenerativeExtractor(CaptionExtractor):
    """A CaptionExtractor that uses a BaseModel to generate captions from signal plots."""

    caption_type = "semantic"

    def __init__(self, config: ChannelConfig, model: BaseModel):
        super().__init__(config)
        self.model = model

    def extract(self, signals: list[Signal]) -> list[tuple[Sample, Annotation]]:
        raise NotImplementedError
