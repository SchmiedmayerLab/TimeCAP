#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from mhc_weekly.constants import WEEKLY_CHANNEL_CONFIG
from timef.schema import Recording
from transformer import Transformer


def _compute_channel_variance(data: np.ndarray) -> np.ndarray:
    variances = np.zeros(data.shape[0], dtype=np.float32)
    for idx, channel in enumerate(data):
        valid = channel[np.isfinite(channel)]
        if valid.size > 0:
            variances[idx] = float(np.var(valid))
    return variances


class WeeklyMHCTransformer(Transformer):
    def __init__(self, config: ChannelConfig = WEEKLY_CHANNEL_CONFIG):
        self.config = config

    def transform_row(self, row: dict) -> Recording:
        user_id = row["user_id"]
        week_start = row["week_start"]
        row_id = f"mhc_weekly:{user_id}:{week_start}"
        data = np.asarray(row["data"], dtype=np.float32).copy()  # (19, 168)

        # Heart rate (channel 5): convert from beats/sec to BPM
        data[5] *= 60
        # Active energy (channel 6): upstream stores mean cal/min over each hour;
        # convert to kcal/hr (* 60 minutes / 1000 cal-per-kcal)
        data[6] *= 60.0 / 1000.0

        n_channels = len(self.config.names)
        if data.shape[0] != n_channels:
            raise ValueError(
                f"data channel count mismatch for {row_id}: expected {n_channels}, got {data.shape[0]}"
            )

        has_any_data = np.array(
            [bool(np.any(~np.isnan(channel))) for channel in data], dtype=bool
        )
        channel_variance = _compute_channel_variance(data)

        display_names = tuple(self.config.display_name(ch) for ch in self.config.names)
        units = tuple(self.config.meta.get(ch, ("", None, 0))[1] for ch in self.config.names)

        return Recording(
            row_id=row_id,
            user_id=user_id,
            date=week_start,
            values=np.ascontiguousarray(data),
            channel_names=tuple(self.config.names),
            display_names=display_names,
            units=units,
            has_any_data=has_any_data,
            channel_variance=channel_variance,
        )
