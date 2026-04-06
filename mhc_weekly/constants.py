#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
import os
import pathlib

from extractors import ChannelConfig
from aggregators import NonZeroAggregator
from detectors.trend import TrendDetector
from detectors.spike import SpikeDetector
from mhc.constants import CHANNEL_NAMES, ACTIVITY_CHANNELS, SLEEP_CHANNELS, CONTINUOUS_CHANNELS

HOURLY_TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent.parent / "templates" / "templates_hourly.json"

WEEKLY_CHANNEL_META = {
    "hk_iphone:HKQuantityTypeIdentifierStepCount":              ("iPhone step count",       "steps/hr",  0),
    "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning": ("iPhone distance",          "m/hr",     1),
    "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed":         ("flights climbed (iPhone)", "count/hr", 1),
    "hk_watch:HKQuantityTypeIdentifierStepCount":               ("Apple Watch step count",   "steps/hr", 0),
    "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning":  ("Apple Watch distance",     "m/hr",     1),
    "hk_watch:HKQuantityTypeIdentifierHeartRate":               ("heart rate",               "bpm",      1),
    "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned":      ("active energy",            "cal/hr",   0),
}

WEEKLY_CHANNEL_CONFIG = ChannelConfig(
    names=CHANNEL_NAMES,
    meta=WEEKLY_CHANNEL_META,
    continuous=CONTINUOUS_CHANNELS,
    templates_path=HOURLY_TEMPLATES_PATH,
    aggregators={"hk_watch:HKQuantityTypeIdentifierHeartRate": NonZeroAggregator()},
    detectors={
        "hk_iphone:HKQuantityTypeIdentifierStepCount":              [TrendDetector(), SpikeDetector()],
        "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning": [TrendDetector(), SpikeDetector()],
        "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed":         [TrendDetector(), SpikeDetector()],
        "hk_watch:HKQuantityTypeIdentifierStepCount":               [TrendDetector(), SpikeDetector()],
        "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning":  [TrendDetector(), SpikeDetector()],
        "hk_watch:HKQuantityTypeIdentifierHeartRate":               [TrendDetector(filter_zeros=True), SpikeDetector(filter_zeros=True)],
        "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned":      [TrendDetector(), SpikeDetector()],
    },
)

WEEKLY_DATASET_DIR = os.environ.get("MHC_WEEKLY_DATASET_DIR", "data/mhc_weekly")
