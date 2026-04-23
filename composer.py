#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from timef.schema import Annotation, CaptionResult, Recording

if TYPE_CHECKING:
    from models.base import BaseModel


REFINED_CAPTION_TYPE = "refined"


DISTILL_PROMPT = """\
Turn the source captions into one coherent, natural time-series annotation.

Work in two internal steps:
1. Distill the source captions by removing duplicate information and merging overlapping facts.
2. Compose the distilled information into a fluent annotation, as if a person wrote it after reading the extracted facts.

Use only the information in the source captions. Do not invent extra values, events, labels, or time ranges.
Keep important channel names, values, and time windows. Make the result sound natural instead of like a list of detector outputs.

Return only the final annotation text. Do not include bullets, headings, explanations, or markdown."""


class Composer:
    """Composes generated captions while preserving global vs. windowed scope."""

    def __init__(self, model: BaseModel, caption_type: str = REFINED_CAPTION_TYPE):
        self.model = model
        self.caption_type = caption_type

    def distill(
        self,
        result: CaptionResult,
        *,
        append: bool = True,
        replace_source: bool = False,
    ) -> list[Annotation]:
        """Distill every row and optionally append or replace row annotations."""
        distilled: list[Annotation] = []
        for row in result.iter_rows():
            annotations = self.distill_recording(row)
            if not annotations:
                continue
            distilled.extend(annotations)
            if replace_source:
                row.annotations = annotations
            elif append:
                row.annotations.extend(annotations)
        return distilled

    def distill_recording(self, row: Recording) -> list[Annotation]:
        """Create coherent annotations without collapsing windowed facts globally."""
        source_annotations = [
            annotation
            for annotation in row.annotations
            if annotation.text and annotation.caption_type != self.caption_type
        ]
        if not source_annotations:
            return []

        distilled: list[Annotation] = []
        for window, annotations in self._annotation_groups(source_annotations):
            annotation = self._distill_group(row, annotations, window=window)
            if annotation is not None:
                distilled.append(annotation)
        return distilled

    def _distill_group(
        self,
        row: Recording,
        source_annotations: list[Annotation],
        *,
        window: tuple[int, int] | None,
    ) -> Annotation | None:
        """Create one coherent annotation for a global or same-window group."""

        formatted_sources = [
            Annotation(
                caption_type=annotation.caption_type,
                text=self._format_source(row, annotation),
                channel_idxs=annotation.channel_idxs,
                window=annotation.window,
                label=annotation.label,
            )
            for annotation in source_annotations
        ]

        response = self.model.process(
            prompt=DISTILL_PROMPT,
            signals=[],
            annotations=formatted_sources,
            multi_channel=True,
        )
        text = self._clean_response(response.text)
        if not text:
            return None

        return Annotation(
            caption_type=self.caption_type,
            text=text,
            channel_idxs=self._source_channel_idxs(row, source_annotations),
            window=window,
        )

    @staticmethod
    def _annotation_groups(
        annotations: list[Annotation],
    ) -> list[tuple[tuple[int, int] | None, list[Annotation]]]:
        groups: dict[tuple[int, int] | None, list[Annotation]] = {}
        for annotation in annotations:
            groups.setdefault(annotation.window, []).append(annotation)
        return [
            (window, groups[window])
            for window in sorted(groups, key=lambda w: (-1, -1) if w is None else w)
        ]

    @staticmethod
    def _format_source(row: Recording, annotation: Annotation) -> str:
        parts = [annotation.caption_type]
        if annotation.channel_idxs:
            names = ", ".join(row.display_names[idx] for idx in annotation.channel_idxs)
            parts.append(f"channels: {names}")
        if annotation.window is not None:
            start, end = annotation.window
            parts.append(f"window: {start}-{end}")
        return f"[{'; '.join(parts)}] {annotation.text}"

    @staticmethod
    def _source_channel_idxs(row: Recording, annotations: list[Annotation]) -> tuple[int, ...]:
        channel_idxs = sorted({idx for annotation in annotations for idx in annotation.channel_idxs})
        if channel_idxs:
            return tuple(channel_idxs)
        return tuple(idx for idx, has_data in enumerate(row.has_any_data) if has_data)

    @staticmethod
    def _clean_response(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:\w+)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"^\s*(?:refined annotation|final annotation)\s*:\s*", "", text, flags=re.I)
        return " ".join(text.split())
