#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import pathlib

from extractors import CaptionExtractor
from timef.schema import Annotation, Recording

TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent.parent / "templates" / "templates.json"
STRUCT_TEMPLATES: dict[str, list[str]] = json.loads(TEMPLATES_PATH.read_text())["structural"]


class StructuralExtractor(CaptionExtractor):
    caption_type = "structural"
    MAX_EVENTS_PER_SIGNAL: int | None = 6

    def extract(self, row: Recording) -> list[Annotation]:
        seed = self._seed(row.row_id)
        results: list[Annotation] = []
        j = 0
        for i, signal in enumerate(row.iter_channels()):
            detectors = self.config.detectors.get(signal.name, [])
            display_name = signal.display_name
            signal_events: list[tuple[float, Annotation]] = []

            for detector in detectors:
                for result in detector.detect(signal.data):
                    templates = STRUCT_TEMPLATES[result.event_type]
                    template = templates[(seed + i + j) % len(templates)]
                    caption = template.format(name=display_name, **result.template_vars())
                    annotation = Annotation(
                        caption_type=self.caption_type,
                        text=caption,
                        channel_idxs=(i,),
                        window=result.window,
                    )
                    signal_events.append((result.score, annotation))
                    j += 1

            signal_events.sort(key=lambda item: item[0], reverse=True)
            if self.MAX_EVENTS_PER_SIGNAL is not None:
                signal_events = signal_events[: self.MAX_EVENTS_PER_SIGNAL]

            results.extend(annotation for _, annotation in signal_events)

        return results
