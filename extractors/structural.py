#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json

from extractors import CaptionExtractor, ChannelConfig
from timef.schema import Annotation, Sample, SampleRef, SampleSignalRef, Signal


class StructuralExtractor(CaptionExtractor):
    caption_type = "structural"

    def __init__(self, config: ChannelConfig):
        super().__init__(config)
        self.templates = json.loads(config.templates_path.read_text())["structural"]

    def extract(self, signals: list[Signal]) -> list[tuple[Sample, Annotation]]:
        seed = self._seed(signals[0].id)
        results = []

        j = 0
        for i, signal in enumerate(signals):
            detectors = self.config.detectors.get(signal.name, [])
            display_name = self.config.meta.get(signal.name, (signal.name, "", 0))[0]

            for detector in detectors:
                for result in detector.detect(signal.data):
                    templates = self.templates[result.event_type]
                    template = templates[(seed + i + j) % len(templates)]
                    caption = template.format(name=display_name, **result.template_vars())

                    sample_id = f"{signal.id}:{self.caption_type}:{j}"
                    sample = Sample(
                        id=sample_id,
                        windows=[result.window],
                        signals=[SampleSignalRef(signal_id=signal.id)],
                    )
                    annotation = Annotation(
                        id=f"{signal.id}:{self.caption_type}:{j}",
                        spec_id=f"captioning:{self.caption_type}",
                        samples=[SampleRef(sample_id=sample_id)],
                        answer=caption,
                    )
                    results.append((sample, annotation))
                    j += 1

        return results
