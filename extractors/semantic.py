#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json

import numpy as np

from extractors import CaptionExtractor, ChannelConfig
from timef.schema import Annotation, Sample, SampleRef, SampleSignalRef, Signal


class SemanticExtractor(CaptionExtractor):
    caption_type = "semantic"

    def __init__(
        self,
        config: ChannelConfig,
        activity_channels: list[str],
        sleep_channels: list[str],
        min_window: int = 5,
    ):
        super().__init__(config)
        templates = json.loads(config.templates_path.read_text())
        self.templates = templates["semantic"]
        self.point_templates = templates["semantic_point"]
        self.activity_channels = activity_channels
        self.sleep_channels = sleep_channels
        self.semantic_names = set(activity_channels) | set(sleep_channels)
        self.min_window = min_window

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
            windows = _contiguous_windows(signal.data, self.min_window)

            for start, end in windows:
                display_name = self.config.display_name(name)
                if end - start <= 1:
                    template = self.point_templates[(seed + j) % len(self.point_templates)]
                    caption = template.format(name=display_name, start=start)
                else:
                    template = self.templates[(seed + j) % len(self.templates)]
                    caption = template.format(name=display_name, start=start, end=end - 1)
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


def _contiguous_windows(data: np.ndarray, min_steps: int) -> list[tuple[int, int]]:
    """Group consecutive non-zero timesteps into (start, end) windows.

    Args:
        data: 1D binary array — one value per timestep.
              Values are 1.0 (event present) or 0.0 (absent).
        min_steps: minimum run length to keep.

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
    keep = lengths >= min_steps
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))
