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
from timef.schema import SignalSpec, Signal
from transformer import Transformer


class MHCTransformer(Transformer):
    def __init__(self, config: ChannelConfig = MHC_CHANNEL_CONFIG):
        self.config = config

    def get_signal_specs(self) -> list[SignalSpec]:
        specs = []
        for ch in self.config.names:
            if ch in self.config.continuous:
                display_name, unit, _ = self.config.meta[ch]
            else:
                display_name = ch
                unit = "binary"
            specs.append(SignalSpec(
                id=f"mhc:{ch}",
                name=ch,
                display_name=display_name,
                channels=["value"],
                unit_sampling_rate="1/min",
                unit_timestamp="minutes",
                unit_value=unit,
            ))
        return specs

    def transform_row(self, row: dict) -> list[Signal]:
        user_id = row["user_id"]
        date = row["date"]
        row_id = f"mhc:{user_id}:{date}"
        data = np.array(row["data"], dtype=np.float32).copy()  # (19, 1440)

        # ZeroToNaNTransform (according to MHC-Benchmark)
        data[5][data[5] == 0] = np.nan
        for ch_idx in (0, 1, 3, 4, 6):
            if np.all(data[ch_idx] == 0):
                data[ch_idx] = np.nan

        # Heart rate (channel 5): convert from beats/sec to BPM
        data[5] *= 60

        total_nonwear_minutes = row.get("total_nonwear_minutes")
        has_any_data = row.get("has_any_data", [])
        minutes_nonzero_or_nan = row.get("minutes_nonzero_or_nan", [])
        channel_names = row.get("channel_names", [])
        channel_units = row.get("channel_units", [])
        channel_variance = row.get("channel_variance", [])

        signals = []
        for i, ch in enumerate(self.config.names):
            signals.append(Signal(
                id=f"mhc:{user_id}:{date}:{ch}",
                spec_id=f"mhc:{ch}",
                name=ch,
                data=data[i],
                metadata={
                    "row_id": row_id,
                    "user_id": user_id,
                    "date": date,
                    "channel_name": channel_names[i] if i < len(channel_names) else ch,
                    "channel_unit": channel_units[i] if i < len(channel_units) else None,
                    "has_any_data": has_any_data[i] if i < len(has_any_data) else None,
                    "minutes_nonzero_or_nan": (
                        minutes_nonzero_or_nan[i] if i < len(minutes_nonzero_or_nan) else None
                    ),
                    "channel_variance": channel_variance[i] if i < len(channel_variance) else None,
                    "total_nonwear_minutes": total_nonwear_minutes,
                },
            ))
        return signals
