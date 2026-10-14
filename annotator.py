#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from extractors import CaptionExtractor
from timef.schema import Annotation, Recording


class Annotator:
    def __init__(self, extractors: list[CaptionExtractor]):
        self.extractors = list(extractors)
        seen = set()
        for extractor in self.extractors:
            if extractor.caption_type in seen:
                raise ValueError(f"Duplicate extractor for caption_type={extractor.caption_type!r}.")
            seen.add(extractor.caption_type)

    def annotate(self, row: Recording) -> list[Annotation]:
        annotations: list[Annotation] = []
        for extractor in self.extractors:
            annotations.extend(extractor.extract(row))
        return annotations
