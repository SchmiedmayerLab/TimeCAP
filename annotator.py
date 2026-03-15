#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from extractors import CaptionExtractor
from timef.schema import Annotation, AnnotationSpec, Sample, Signal


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

    def annotate(self, signals: list[Signal]) -> tuple[list[Sample], list[Annotation]]:
        samples: list[Sample] = []
        annotations: list[Annotation] = []
        for extractor in self.extractors:
            for s, a in extractor.extract(signals):
                samples.append(s)
                annotations.append(a)
        return samples, annotations
