#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
import os
import pathlib

from extractors import ChannelConfig
from aggregators import NonZeroAggregator
from detectors.trend import TrendDetector
from detectors.spike import SpikeDetector
from mhc.constants import ACTIVITY_CHANNELS, CHANNEL_NAMES, CONTINUOUS_CHANNELS, SLEEP_CHANNELS

HOURLY_TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent.parent / "templates" / "templates_hourly.json"

WEEKLY_CHANNEL_META = {
    "hk_iphone:HKQuantityTypeIdentifierStepCount":              ("iPhone step count",       "steps/hr",  0),
    "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning": ("iPhone distance",          "m/hr",     1),
    "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed":         ("flights climbed (iPhone)", "count/hr", 1),
    "hk_watch:HKQuantityTypeIdentifierStepCount":               ("Apple Watch step count",   "steps/hr", 0),
    "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning":  ("Apple Watch distance",     "m/hr",     1),
    "hk_watch:HKQuantityTypeIdentifierHeartRate":               ("heart rate",               "bpm",      1),
    "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned":      ("active energy",            "kcal/hr",  0),
}

WEEKLY_CHANNEL_CONFIG = ChannelConfig(
    names=CHANNEL_NAMES,
    meta=WEEKLY_CHANNEL_META,
    continuous=CONTINUOUS_CHANNELS,
    groups={
        "activity": frozenset(ACTIVITY_CHANNELS),
        "sleep": frozenset(SLEEP_CHANNELS),
    },
    templates_path=HOURLY_TEMPLATES_PATH,
    time_unit="hours",
    aggregators={"hk_watch:HKQuantityTypeIdentifierHeartRate": NonZeroAggregator()},
    detectors={
        "hk_iphone:HKQuantityTypeIdentifierStepCount":              [TrendDetector(), SpikeDetector(min_height=1000.0, min_distance=1, top_k=10)],
        "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning": [TrendDetector(), SpikeDetector(min_height=720.0, min_distance=1, top_k=10)],
        "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed":         [TrendDetector(), SpikeDetector(min_height=1.0, min_distance=1, top_k=10)],
        "hk_watch:HKQuantityTypeIdentifierStepCount":               [TrendDetector(), SpikeDetector(min_height=1000.0, min_distance=1, top_k=10)],
        "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning":  [TrendDetector(), SpikeDetector(min_height=720.0, min_distance=1, top_k=10)],
        "hk_watch:HKQuantityTypeIdentifierHeartRate":               [TrendDetector(filter_zeros=True), SpikeDetector(filter_zeros=True, min_height=100.0, min_prominence=5.0, min_distance=1, top_k=10)],
        "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned":      [TrendDetector(), SpikeDetector(min_height=30.0, min_distance=1, top_k=10)],
    },
)

WEEKLY_DATASET_DIR = os.environ.get("MHC_WEEKLY_DATASET_DIR", "data/mhc_weekly")
