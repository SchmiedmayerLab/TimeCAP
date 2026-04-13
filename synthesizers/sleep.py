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
from synthesizers import CrossChannelSynthesizer, contiguous_windows
from timef.schema import Annotation, Recording


class SleepSynthesizer(CrossChannelSynthesizer):
    def __init__(self, min_duration: int = 0):
        self.min_duration = min_duration

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        try:
            in_bed_idx = row.channel_names.index("sleep:inbed")
            asleep_idx = row.channel_names.index("sleep:asleep")
        except ValueError:
            return []

        in_bed = np.asarray(row.values[in_bed_idx], dtype=float)
        asleep = np.asarray(row.values[asleep_idx], dtype=float)
        if not np.any(~np.isnan(asleep)):
            return []
        mask = (~np.isnan(in_bed)) & (in_bed > 0) & ~((~np.isnan(asleep)) & (asleep > 0))

        time_unit = "hour" if config.time_unit == "hours" else "minute"
        templates = json.loads(config.templates_path.read_text())["cross_channel"]["sleep"]
        seed = CaptionExtractor._seed(row.row_id)

        results: list[Annotation] = []
        for i, (start, end) in enumerate(contiguous_windows(mask, self.min_duration)):
            end_inclusive = max(start, end - 1)
            template = templates[(seed + i) % len(templates)]
            text = template.format(time_unit=time_unit, start=start, end=end_inclusive)
            results.append(
                Annotation(
                    caption_type="cross_channel",
                    text=text,
                    channel_idxs=(asleep_idx, in_bed_idx),
                    window=(start, end),
                    label="in_bed_not_sleeping",
                )
            )
        return results
