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
from timef.schema import Annotation, RuntimeRow

TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent.parent / "templates" / "templates.json"
STRUCT_TEMPLATES: dict[str, list[str]] = json.loads(TEMPLATES_PATH.read_text())["structural"]


class StructuralExtractor(CaptionExtractor):
    caption_type = "structural"

    def extract(self, row: RuntimeRow) -> list[Annotation]:
        seed = self._seed(row.row_id)
        results: list[Annotation] = []
        j = 0
        for i, signal in enumerate(row.iter_channels()):
            detectors = self.config.detectors.get(signal.name, [])
            display_name = signal.display_name

            for detector in detectors:
                for result in detector.detect(signal.data):
                    templates = STRUCT_TEMPLATES[result.event_type]
                    template = templates[(seed + i + j) % len(templates)]
                    caption = template.format(name=display_name, **result.template_vars())
                    results.append(
                        Annotation(
                            caption_type=self.caption_type,
                            text=caption,
                            channel_idxs=(i,),
                            window=result.window,
                        )
                    )
                    j += 1

        return results
