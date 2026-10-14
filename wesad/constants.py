#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import os
from pathlib import Path

from aggregators import NonZeroAggregator
from detectors.spike import SpikeDetector
from detectors.trend import TrendDetector
from extractors import ChannelConfig

WESAD_CHANNEL_NAMES = [
    "wesad:chest:acc_x",
    "wesad:chest:acc_y",
    "wesad:chest:acc_z",
    "wesad:chest:acc_mag",
    "wesad:chest:emg",
    "wesad:chest:eda",
    "wesad:chest:temp",
    "wesad:chest:resp",
    "wesad:wrist:acc_x",
    "wesad:wrist:acc_y",
    "wesad:wrist:acc_z",
    "wesad:wrist:acc_mag",
    "wesad:wrist:bvp",
    "wesad:wrist:eda",
    "wesad:wrist:temp",
    "wesad:e4:hr",
    "wesad:label:baseline",
    "wesad:label:stress",
    "wesad:label:amusement",
    "wesad:label:meditation",
]

WESAD_CHANNEL_META = {
    "wesad:chest:acc_x": ("chest acc x", "g", 3),
    "wesad:chest:acc_y": ("chest acc y", "g", 3),
    "wesad:chest:acc_z": ("chest acc z", "g", 3),
    "wesad:chest:acc_mag": ("chest acc magnitude", "g", 3),
    "wesad:chest:emg": ("chest emg", "mV", 3),
    "wesad:chest:eda": ("chest eda", "uS", 3),
    "wesad:chest:temp": ("chest temperature", "degC", 2),
    "wesad:chest:resp": ("chest respiration magnitude", "a.u.", 3),
    "wesad:wrist:acc_x": ("wrist acc x", "g", 3),
    "wesad:wrist:acc_y": ("wrist acc y", "g", 3),
    "wesad:wrist:acc_z": ("wrist acc z", "g", 3),
    "wesad:wrist:acc_mag": ("wrist acc magnitude", "g", 3),
    "wesad:wrist:bvp": ("wrist bvp magnitude", "a.u.", 3),
    "wesad:wrist:eda": ("wrist eda", "uS", 3),
    "wesad:wrist:temp": ("wrist temperature", "degC", 2),
    "wesad:e4:hr": ("wrist heart rate", "bpm", 2),
    "wesad:label:baseline": ("baseline state", "binary", 0),
    "wesad:label:stress": ("stress state", "binary", 0),
    "wesad:label:amusement": ("amusement state", "binary", 0),
    "wesad:label:meditation": ("meditation state", "binary", 0),
}

WESAD_CONTINUOUS_CHANNELS = frozenset(
    name for name in WESAD_CHANNEL_NAMES if not name.startswith("wesad:label:")
)

CHEST_SAMPLE_RATE_HZ = 700

WESAD_PRIMARY_LABEL_CHANNELS: dict[int, str] = {
    1: "wesad:label:baseline",
    2: "wesad:label:stress",
    3: "wesad:label:amusement",
    4: "wesad:label:meditation",
}

WESAD_PRIMARY_STATE_CHANNELS = frozenset(WESAD_PRIMARY_LABEL_CHANNELS.values())

WESAD_LABEL_CHANNELS = frozenset(
    name for name in WESAD_CHANNEL_NAMES if name.startswith("wesad:label:")
)

WESAD_HR_CHANNEL = "wesad:e4:hr"
WESAD_CHEST_EDA_CHANNEL = "wesad:chest:eda"
WESAD_WRIST_EDA_CHANNEL = "wesad:wrist:eda"
WESAD_CHEST_RESP_CHANNEL = "wesad:chest:resp"
WESAD_CHEST_ACC_MAG_CHANNEL = "wesad:chest:acc_mag"
WESAD_WRIST_ACC_MAG_CHANNEL = "wesad:wrist:acc_mag"

WESAD_AGGREGATORS = {
    "wesad:e4:hr": NonZeroAggregator(),
}

_TREND_ACC_PARAMS = {
    "min_window": 40,
    "max_window": 150,
    "min_effect": 1.8,
    "min_r2": 0.4,
    "min_span": 30,
    "merge_gap": 15,
}

_SPIKE_CHEST_ACC_MAG_PARAMS = {
    "min_prominence": 0.05,
    "min_distance": 60,
    "top_k": 4,
}

_SPIKE_WRIST_ACC_MAG_PARAMS = {
    "min_prominence": 0.1,
    "min_distance": 60,
    "top_k": 4,
}

_TREND_TEMP_PARAMS = {
    "min_window": 90,
    "max_window": 600,
    "min_effect": 0.18,
    "min_r2": 0.70,
    "min_span": 60,
    "merge_gap": 30,
}

_TREND_EDA_PARAMS = {
    "min_window": 30,
    "max_window": 240,
    "min_effect": 0.10,
    "min_r2": 0.60,
    "min_span": 40,
    "merge_gap": 15,
}

_SPIKE_TEMP_PARAMS = {
    "min_prominence": 0.25,
    "min_distance": 60,
    "top_k": 4,
}

WESAD_DETECTORS = {
    "wesad:chest:acc_mag": [
        TrendDetector(**_TREND_ACC_PARAMS),
        SpikeDetector(**_SPIKE_CHEST_ACC_MAG_PARAMS),
    ],
    "wesad:chest:emg": [
        TrendDetector(
            min_window=20,
            max_window=240,
            min_effect=0.6,
            min_r2=0.15,
            min_span=14,
            merge_gap=15,
        ),
    ],
    "wesad:e4:hr": [
        TrendDetector(filter_zeros=True),
        SpikeDetector(
            filter_zeros=True,
            min_height=100.0,
            min_prominence=5.0,
            min_distance=60,
            min_width=1,
            top_k=4,
        ),
    ],
    "wesad:chest:eda": [
        TrendDetector(**_TREND_EDA_PARAMS),
        SpikeDetector(
            min_prominence=0.25,
            min_distance=60,
            min_width=1,
            top_k=4,
        ),
    ],
    "wesad:wrist:eda": [
        TrendDetector(**_TREND_EDA_PARAMS),
        SpikeDetector(
            min_prominence=0.5,
            min_distance=60,
            min_width=1,
            top_k=4,
        ),
    ],
    "wesad:chest:temp": [
        TrendDetector(**_TREND_TEMP_PARAMS),
        SpikeDetector(**_SPIKE_TEMP_PARAMS),
    ],
    "wesad:wrist:temp": [
        TrendDetector(**_TREND_TEMP_PARAMS),
        SpikeDetector(**_SPIKE_TEMP_PARAMS),
    ],
    "wesad:chest:resp": [
        TrendDetector(
            min_window=24,
            max_window=180,
            min_effect=0.75,
            min_r2=0.2,
            min_span=16,
            merge_gap=15,
        ),
        SpikeDetector(
            min_prominence=10.0,
            min_distance=60,
            min_width=1,
            top_k=4,
        ),
    ],
    "wesad:wrist:acc_mag": [
        TrendDetector(**_TREND_ACC_PARAMS),
        SpikeDetector(**_SPIKE_WRIST_ACC_MAG_PARAMS),
    ],
    "wesad:wrist:bvp": [
        TrendDetector(
            min_window=20,
            max_window=300,
            min_effect=0.7,
            min_r2=0.15,
            min_span=14,
            merge_gap=15,
        ),
        SpikeDetector(
            min_prominence=200.0,
            min_distance=60,
            min_width=1,
            top_k=4,
        ),
    ],
}

WESAD_TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "templates" / "templates_wesad.json"

try:
    WESAD_CHANNEL_CONFIG = ChannelConfig(
        names=WESAD_CHANNEL_NAMES,
        meta=WESAD_CHANNEL_META,
        continuous=WESAD_CONTINUOUS_CHANNELS,
        groups={
            "activity": WESAD_PRIMARY_STATE_CHANNELS,
        },
        aggregators=WESAD_AGGREGATORS,
        detectors=WESAD_DETECTORS,
        templates_path=WESAD_TEMPLATES_PATH,
        time_unit="seconds",
    )
except Exception as e:
    raise RuntimeError(
        "failed to build WESAD_CHANNEL_CONFIG "
        f"(templates_path={WESAD_TEMPLATES_PATH}, "
        f"names={len(WESAD_CHANNEL_NAMES)}, meta={len(WESAD_CHANNEL_META)}, "
        f"continuous={len(WESAD_CONTINUOUS_CHANNELS)})"
    ) from e

TIMEF_DATASET_DIR = os.environ.get("WESAD_TIMEF_DATASET_DIR", "data/wesad_timef")
