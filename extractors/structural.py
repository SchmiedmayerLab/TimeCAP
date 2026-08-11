#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json

from extractors import CaptionExtractor, ChannelConfig
from timef.schema import Annotation, Recording
from util import capitalize


class StructuralExtractor(CaptionExtractor):
    caption_type = "structural"

    def __init__(self, config: ChannelConfig):
        super().__init__(config)
        self.templates: dict[str, list[str]] = json.loads(config.templates_path.read_text())["structural"]

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
                    templates = self.templates[result.event_type]
                    template = templates[(seed + i + j) % len(templates)]
                    caption = capitalize(template.format(name=display_name, **result.template_vars()))
                    annotation = Annotation(
                        caption_type=self.caption_type,
                        text=caption,
                        channel_idxs=(i,),
                        window=result.window,
                    )
                    signal_events.append((result.score, annotation))
                    j += 1

            signal_events.sort(key=lambda item: item[0], reverse=True)

            results.extend(annotation for _, annotation in signal_events)

        return results
