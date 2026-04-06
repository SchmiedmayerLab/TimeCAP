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
from timef.schema import Annotation, Recording

TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent.parent / "templates" / "templates.json"
SEM_TEMPLATES: list[str] = json.loads(TEMPLATES_PATH.read_text())["semantic"]


class SemanticExtractor(CaptionExtractor):
    caption_type = "semantic"

    def __init__(self, config: ChannelConfig, min_minutes: int = 5):
        super().__init__(config)
        self.activity_channels = self.config.groups.get("activity", frozenset())
        self.sleep_channels = self.config.groups.get("sleep", frozenset())
        self.semantic_names = self.activity_channels | self.sleep_channels
        self.min_minutes = min_minutes

    def extract(self, row: Recording) -> list[Annotation]:
        seed = self._seed(row.row_id)
        semantic_indices = [i for i, name in enumerate(row.channel_names) if name in self.semantic_names]
        if not semantic_indices:
            return []

        continuous_indices = tuple(
            i for i, name in enumerate(row.channel_names) if name in self.config.continuous
        )

        results: list[Annotation] = []
        j = 0
        for idx in semantic_indices:
            name = row.channel_names[idx]
            windows = _contiguous_windows(row.values[idx], self.min_minutes)

            for start, end in windows:
                template = SEM_TEMPLATES[(seed + j) % len(SEM_TEMPLATES)]
                display_name = self.config.display_name(name)
                caption = template.format(name=display_name, start=start, end=max(end - 1, start))
                caption = caption[0].upper() + caption[1:]
                label = name if name in self.activity_channels else None
                results.append(
                    Annotation(
                        caption_type=self.caption_type,
                        text=caption,
                        channel_idxs=continuous_indices,
                        window=(start, end),
                        label=label,
                    )
                )
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
