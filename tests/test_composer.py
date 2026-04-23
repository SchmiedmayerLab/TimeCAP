#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np

from composer import Composer
from timef.schema import Annotation, CaptionResult, Recording


class FakeModel:
    def __init__(self):
        self.captions: list[str] = []

    def process(self, prompt, signals, annotations=None, multi_channel=False):
        caption = "\n".join(annotation.text for annotation in annotations or [])
        self.captions.append(caption)
        return type("Response", (), {"text": f"Refined annotation: composed {len(self.captions)}"})()


def test_composer_distills_global_and_windowed_annotations_separately():
    model = FakeModel()
    row = _recording()
    row.annotations = [
        Annotation("statistical", "Heart rate has a daily mean of 80 bpm.", (0,)),
        Annotation("structural", "Heart rate rises from minute 0 to 2.", (0,), window=(0, 2)),
        Annotation("semantic", "Sleep is active from minute 0 to 2.", (1,), window=(0, 2)),
    ]

    annotations = Composer(model).distill_recording(row)

    assert len(annotations) == 2
    assert annotations[0].caption_type == "refined"
    assert annotations[0].channel_idxs == (0,)
    assert annotations[0].window is None
    assert annotations[0].text == "composed 1"
    assert annotations[1].caption_type == "refined"
    assert annotations[1].channel_idxs == (0, 1)
    assert annotations[1].window == (0, 2)
    assert annotations[1].text == "composed 2"
    assert "[statistical; channels: Heart Rate]" in model.captions[0]
    assert "[structural; channels: Heart Rate; window: 0-2]" in model.captions[1]
    assert "[semantic; channels: Sleep; window: 0-2]" in model.captions[1]


def test_composer_can_replace_source_annotations():
    row = _recording()
    row.annotations = [Annotation("structural", "Heart rate rises from minute 0 to 2.", (0,))]
    result = CaptionResult(rows=[row])

    refined = Composer(FakeModel()).distill(result, replace_source=True)

    assert refined == row.annotations
    assert len(row.annotations) == 1
    assert row.annotations[0].caption_type == "refined"


def _recording() -> Recording:
    return Recording(
        row_id="r1",
        user_id="u1",
        date="2026-01-01",
        values=np.zeros((2, 3)),
        channel_names=("hr", "sleep"),
        display_names=("Heart Rate", "Sleep"),
        units=("bpm", None),
        has_any_data=np.array([True, True]),
        channel_variance=np.array([1.0, 0.0]),
    )
