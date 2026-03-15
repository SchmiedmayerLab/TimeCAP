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
        data = np.array(row["data"], dtype=np.float32)  # (19, 1440)
        nonwear = row["nonwear"]                         # (1440,) uint8

        # Mask nonwear minutes as NaN
        data = data.copy()
        data[:, nonwear != 0] = np.nan

        signals = []
        for i, ch in enumerate(self.config.names):
            signals.append(Signal(
                id=f"mhc:{user_id}:{date}:{ch}",
                spec_id=f"mhc:{ch}",
                name=ch,
                data=data[i],
                metadata={"user_id": user_id, "date": date},
            ))
        return signals
