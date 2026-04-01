#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from extractors import CaptionExtractor
from timef.runtime import CaptionData, SignalData
from timef.schema import AnnotationSpec


class Annotator:
    def __init__(self, extractors: list[CaptionExtractor]):
        self.extractors = list(extractors)
        seen = set()
        for extractor in self.extractors:
            if extractor.caption_type in seen:
                raise ValueError(f"Duplicate extractor for caption_type={extractor.caption_type!r}.")
            seen.add(extractor.caption_type)

    def get_annotation_specs(self) -> list[AnnotationSpec]:
        return [extractor.get_annotation_spec() for extractor in self.extractors]

    def annotate(self, signals: list[SignalData]) -> list[CaptionData]:
        results: list[CaptionData] = []
        for extractor in self.extractors:
            results.extend(extractor.extract(signals))
        return results
