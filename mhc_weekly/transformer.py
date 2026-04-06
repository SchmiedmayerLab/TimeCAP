#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from mhc_weekly.constants import WEEKLY_CHANNEL_CONFIG
from timef.schema import SignalSpec, Signal
from transformer import Transformer


class WeeklyMHCTransformer(Transformer):
    def __init__(self, config: ChannelConfig = WEEKLY_CHANNEL_CONFIG):
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
                id=f"mhc_weekly:{ch}",
                name=ch,
                display_name=display_name,
                channels=["value"],
                unit_sampling_rate="1/hour",
                unit_timestamp="hours",
                unit_value=unit,
            ))
        return specs

    def transform_row(self, row: dict) -> list[Signal]:
        user_id = row["user_id"]
        week_start = row["week_start"]
        row_id = f"mhc_weekly:{user_id}:{week_start}"
        data = np.array(row["data"], dtype=np.float32).copy()  # (19, 168)

        # Heart rate (channel 5): convert from beats/sec to BPM
        data[5] *= 60

        signals = []
        for i, ch in enumerate(self.config.names):
            signals.append(Signal(
                id=f"mhc_weekly:{user_id}:{week_start}:{ch}",
                spec_id=f"mhc_weekly:{ch}",
                name=ch,
                data=data[i],
                metadata={
                    "row_id": row_id,
                    "user_id": user_id,
                    "week_start": week_start,
                    "unit_timestamp": "hours",
                    "n_valid_days": row.get("n_valid_days"),
                    "n_valid_hours": row.get("n_valid_hours"),
                    "n_data_days": row.get("n_data_days"),
                },
            ))
        return signals
