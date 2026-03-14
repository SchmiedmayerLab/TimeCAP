from __future__ import annotations

import json
import pathlib
from typing import Dict, List

from extractors import CaptionExtractor
from timef.schema import Annotation, Sample, SampleRef, SampleSignalRef, Signal

TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent.parent / "templates" / "templates.json"
STRUCT_TEMPLATES: Dict[str, List[str]] = json.loads(TEMPLATES_PATH.read_text())["structural"]


class StructuralExtractor(CaptionExtractor):
    caption_type = "structural"

    def extract(self, signals: list[Signal]) -> list[tuple[Sample, Annotation]]:
        seed = hash(signals[0].id)
        results = []

        j = 0
        for i, signal in enumerate(signals):
            detectors = self.config.detectors.get(signal.name, [])
            display_name = self.config.meta.get(signal.name, (signal.name, "", 0))[0]

            for detector in detectors:
                for result in detector.detect(signal.data):
                    templates = STRUCT_TEMPLATES[result.event_type]
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
