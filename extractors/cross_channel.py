#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np

from extractors import CaptionExtractor, ChannelConfig
from timef.schema import Annotation, Recording


class CrossChannelExtractor(CaptionExtractor):
    caption_type = "cross_channel"

    def __init__(self, config: ChannelConfig, min_duration: int = 5):
        super().__init__(config)
        self.min_duration = min_duration

    def extract(self, row: Recording) -> list[Annotation]:
        try:
            in_bed_idx = row.channel_names.index("sleep:inbed")
            asleep_idx = row.channel_names.index("sleep:asleep")
        except ValueError:
            return []

        in_bed = np.asarray(row.values[in_bed_idx], dtype=float)
        asleep = np.asarray(row.values[asleep_idx], dtype=float)
        if not np.any((~np.isnan(asleep)) & (asleep > 0)):
            return []
        mask = (~np.isnan(in_bed)) & (in_bed > 0) & ~((~np.isnan(asleep)) & (asleep > 0))

        results: list[Annotation] = []
        for start, end in _contiguous_windows(mask, self.min_duration):
            end_inclusive = max(start, end - 1)
            results.append(
                Annotation(
                    caption_type=self.caption_type,
                    text=f"In bed but not sleeping from {self._time_label(start)} {start} to {self._time_label(end_inclusive)} {end_inclusive}.",
                    channel_idxs=(asleep_idx, in_bed_idx),
                    window=(start, end),
                    label="in_bed_not_sleeping",
                )
            )
        return results

    def _time_label(self, value: int) -> str:
        return "hour" if self.config.time_unit == "hours" else "minute"


def _contiguous_windows(mask: np.ndarray, min_duration: int) -> list[tuple[int, int]]:
    if not mask.any():
        return []

    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    keep = (ends - starts) >= min_duration
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))
