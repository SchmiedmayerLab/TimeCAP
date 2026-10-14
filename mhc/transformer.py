#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from mhc.constants import MHC_CHANNEL_CONFIG
from timef.schema import Recording
from transformer import Transformer


def _validate_channel_length(field_name: str, values, expected: int, row_id: str) -> None:
    actual = len(values)
    if actual != expected:
        raise ValueError(
            f"{field_name} length mismatch for {row_id}: expected {expected}, got {actual}"
        )


def _compute_channel_variance(data: np.ndarray) -> np.ndarray:
    variances = np.zeros(data.shape[0], dtype=np.float32)
    for idx, channel in enumerate(data):
        valid = channel[np.isfinite(channel)]
        if valid.size > 0:
            variances[idx] = float(np.var(valid))
    return variances


class MHCTransformer(Transformer):
    def __init__(self, config: ChannelConfig = MHC_CHANNEL_CONFIG):
        self.config = config

    def transform_row(self, row: dict) -> Recording:
        user_id = row["user_id"]
        date = row["date"]
        row_id = f"mhc:{user_id}:{date}"
        data = np.asarray(row["data"], dtype=np.float32).copy()  # (19, 1440)

        # Treat placeholder zero-only values as missing data.
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
        n_channels = len(self.config.names)
        if data.shape[0] != n_channels:
            raise ValueError(
                f"data channel count mismatch for {row_id}: expected {n_channels}, got {data.shape[0]}"
            )
        _validate_channel_length("has_any_data", has_any_data, n_channels, row_id)
        _validate_channel_length("minutes_nonzero_or_nan", minutes_nonzero_or_nan, n_channels, row_id)
        _validate_channel_length("channel_names", channel_names, n_channels, row_id)
        _validate_channel_length("channel_units", channel_units, n_channels, row_id)

        # The stored dataset metadata only provides variance for a subset of channels.
        # Recompute per-channel variance from the transformed data so the runtime
        # representation remains aligned with the actual signal matrix.
        channel_variance = _compute_channel_variance(data)

        display_names = []
        units = []
        for i, ch in enumerate(self.config.names):
            display_names.append(self.config.display_name(ch))
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
