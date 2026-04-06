#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from mhc.constants import MHC_CHANNEL_CONFIG
from timef.schema import Recording
from transformer import Transformer


class MHCTransformer(Transformer):
    def __init__(self, config: ChannelConfig = MHC_CHANNEL_CONFIG):
        self.config = config

    def transform_row(self, row: dict) -> Recording:
        user_id = row["user_id"]
        date = row["date"]
        row_id = f"mhc:{user_id}:{date}"
        data = np.asarray(row["data"], dtype=np.float32).copy()  # (19, 1440)

        # ZeroToNaNTransform (according to MHC-Benchmark)
        data[5][data[5] == 0] = np.nan
        for ch_idx in (0, 1, 3, 4, 6):
            if np.all(data[ch_idx] == 0):
                data[ch_idx] = np.nan

        # Heart rate (channel 5): convert from beats/sec to BPM
        data[5] *= 60

        has_any_data = np.asarray(row.get("has_any_data", []), dtype=bool)
        minutes_nonzero_or_nan = np.asarray(row.get("minutes_nonzero_or_nan", []), dtype=np.float32)
        channel_names = row.get("channel_names", [])
        channel_units = row.get("channel_units", [])
        channel_variance = np.asarray(row.get("channel_variance", []), dtype=np.float32)

        n_channels = len(self.config.names)
        if has_any_data.shape != (n_channels,):
            has_any_data = np.array(
                [has_any_data[i] if i < len(has_any_data) else False for i in range(n_channels)],
                dtype=bool,
            )
        if minutes_nonzero_or_nan.shape != (n_channels,):
            minutes_nonzero_or_nan = np.array(
                [
                    float(minutes_nonzero_or_nan[i]) if i < len(minutes_nonzero_or_nan) else 0.0
                    for i in range(n_channels)
                ],
                dtype=np.float32,
            )
        if channel_variance.shape != (n_channels,):
            channel_variance = np.array(
                [float(channel_variance[i]) if i < len(channel_variance) else 0.0 for i in range(n_channels)],
                dtype=np.float32,
            )

        display_names = []
        units = []
        for i, ch in enumerate(self.config.names):
            display_names.append(channel_names[i] if i < len(channel_names) else self.config.display_name(ch))
            units.append(channel_units[i] if i < len(channel_units) else self.config.meta.get(ch, ("", None, 0))[1])

        return Recording(
            row_id=row_id,
            user_id=user_id,
            date=date,
            values=np.ascontiguousarray(data),
            channel_names=tuple(self.config.names),
            display_names=tuple(display_names),
            units=tuple(units),
            has_any_data=has_any_data,
            minutes_nonzero_or_nan=minutes_nonzero_or_nan,
            channel_variance=channel_variance,
            total_nonwear_minutes=row.get("total_nonwear_minutes"),
            wear_pct=row.get("wear_pct"),
        )
