#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import pathlib

import numpy as np

from extractors import CaptionExtractor, ChannelConfig
from timef.schema import Annotation, Sample, SampleRef, SampleSignalRef, Signal

TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent.parent / "templates" / "templates.json"
SEM_TEMPLATES: list[str] = json.loads(TEMPLATES_PATH.read_text())["semantic"]


class SemanticExtractor(CaptionExtractor):
    caption_type = "semantic"

    def __init__(
        self,
        config: ChannelConfig,
        activity_channels: list[str],
        sleep_channels: list[str],
        min_minutes: int = 5,
    ):
        super().__init__(config)
        self.activity_channels = activity_channels
        self.sleep_channels = sleep_channels
        self.semantic_names = set(activity_channels) | set(sleep_channels)
        self.min_minutes = min_minutes

    def extract(self, signals: list[Signal]) -> list[tuple[Sample, Annotation]]:
        seed = self._seed(signals[0].id)
        semantic_signals = {s.name: s for s in signals if s.name in self.semantic_names}
        if not semantic_signals:
            return []

        row_key = signals[0].id.rsplit(":", 1)[0]

        continuous_refs = [
            SampleSignalRef(signal_id=s.id)
            for s in signals if s.name in self.config.continuous
        ]

        results = []
        j = 0
        for name, signal in semantic_signals.items():
            windows = _contiguous_windows(signal.data, self.min_minutes)

            for start, end in windows:
                template = SEM_TEMPLATES[(seed + j) % len(SEM_TEMPLATES)]
                display_name = self.config.display_name(name)
                caption = template.format(name=display_name, start=start, end=max(end - 1, start))
                caption = caption[0].upper() + caption[1:]

                label = name if name in self.activity_channels else None

                sample_id = f"{row_key}:{self.caption_type}:{name}:{j}"
                sample = Sample(
                    id=sample_id,
                    windows=[(start, end)],
                    signals=continuous_refs,
                )
                annotation = Annotation(
                    id=sample_id,
                    spec_id=f"captioning:{self.caption_type}",
                    samples=[SampleRef(sample_id=sample_id)],
                    answer=caption,
                    label=label,
                )
                results.append((sample, annotation))
                j += 1

        return results


def _contiguous_windows(data: np.ndarray, min_minutes: int) -> list[tuple[int, int]]:
    """Group consecutive non-zero minutes into (start, end) windows.

    Args:
        data: 1D binary array of shape (1440,) — one value per minute.
              Values are 1.0 (event present) or 0.0 (absent).
        min_minutes: minimum run length to keep.

    Returns:
        List of (start, end) tuples where end is exclusive.
    """
    mask = (~np.isnan(data)) & (data > 0)
    if not mask.any():
        return []

    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    lengths = ends - starts
    keep = lengths >= min_minutes
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))
