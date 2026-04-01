#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import pathlib

from extractors import CAPTION_SPEC_IDS, CaptionExtractor
from timef.runtime import CaptionData, SignalData

TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent.parent / "templates" / "templates.json"
STRUCT_TEMPLATES: dict[str, list[str]] = json.loads(TEMPLATES_PATH.read_text())["structural"]


class StructuralExtractor(CaptionExtractor):
    caption_type = "structural"

    def extract(self, signals: list[SignalData]) -> list[CaptionData]:
        if not signals:
            return []
        seed = self._seed(signals[0].id)
        results = []

        j = 0
        for i, signal in enumerate(signals):
            detectors = self.config.detectors.get(signal.name, [])
            display_name = self.config.meta.get(signal.name, (signal.name, "", 0))[0]

            for detector in detectors:
                for detection in detector.detect(signal.data):
                    templates = STRUCT_TEMPLATES[detection.event_type]
                    template = templates[(seed + i + j) % len(templates)]
                    caption = template.format(name=display_name, **detection.template_vars())

                    sample_id = f"{signal.id}:{self.caption_type}:{j}"
                    results.append(CaptionData(
                        signal_id=signal.id,
                        sample_id=sample_id,
                        spec_id=CAPTION_SPEC_IDS[self.caption_type],
                        answer=caption,
                        window=detection.window,
                    ))
                    j += 1

        return results
