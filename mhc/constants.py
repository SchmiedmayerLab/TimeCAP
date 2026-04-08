#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
import os

from extractors import ChannelConfig
from aggregators import NonZeroAggregator
from detectors.trend import TrendDetector
from detectors.spike import SpikeDetector

CHANNEL_NAMES = [
    "hk_iphone:HKQuantityTypeIdentifierStepCount",
    "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning",
    "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed",
    "hk_watch:HKQuantityTypeIdentifierStepCount",
    "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning",
    "hk_watch:HKQuantityTypeIdentifierHeartRate",
    "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned",
    "sleep:asleep", "sleep:inbed",
    "workout:HKWorkoutActivityTypeWalking",
    "workout:HKWorkoutActivityTypeCycling",
    "workout:HKWorkoutActivityTypeRunning",
    "workout:HKWorkoutActivityTypeOther",
    "workout:HKWorkoutActivityTypeMixedMetabolicCardioTraining",
    "workout:HKWorkoutActivityTypeTraditionalStrengthTraining",
    "workout:HKWorkoutActivityTypeElliptical",
    "workout:HKWorkoutActivityTypeHighIntensityIntervalTraining",
    "workout:HKWorkoutActivityTypeFunctionalStrengthTraining",
    "workout:HKWorkoutActivityTypeYoga",
]

CHANNEL_META = {
    "hk_iphone:HKQuantityTypeIdentifierStepCount":              ("iPhone step count",       "steps/min",  1),
    "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning": ("iPhone distance",          "m/min",     3),
    "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed":         ("flights climbed (iPhone)", "count/min", 1),
    "hk_watch:HKQuantityTypeIdentifierStepCount":               ("Apple Watch step count",   "steps/min", 1),
    "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning":  ("Apple Watch distance",     "m/min",     3),
    "hk_watch:HKQuantityTypeIdentifierHeartRate":               ("heart rate",               "bpm",       1),
    "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned":      ("active energy",            "cal/min",   1),
}

CONTINUOUS_CHANNELS = frozenset(CHANNEL_META.keys())

ACTIVITY_CHANNELS = [
    "workout:HKWorkoutActivityTypeWalking",
    "workout:HKWorkoutActivityTypeCycling",
    "workout:HKWorkoutActivityTypeRunning",
    "workout:HKWorkoutActivityTypeOther",
    "workout:HKWorkoutActivityTypeMixedMetabolicCardioTraining",
    "workout:HKWorkoutActivityTypeTraditionalStrengthTraining",
    "workout:HKWorkoutActivityTypeElliptical",
    "workout:HKWorkoutActivityTypeHighIntensityIntervalTraining",
    "workout:HKWorkoutActivityTypeFunctionalStrengthTraining",
    "workout:HKWorkoutActivityTypeYoga",
]

SLEEP_CHANNELS = ["sleep:asleep", "sleep:inbed"]


def _activity_spike(**overrides) -> SpikeDetector:
    params = {
        "prominence_scale": 5.0,
        "min_distance": 30,
        "min_width": 3,
        "top_k": 4,
        "smooth_window": 5,
        "drop_localization": "left_edge",
    }
    params.update(overrides)
    return SpikeDetector(**params)


def _heart_rate_spike() -> SpikeDetector:
    return SpikeDetector(
        filter_zeros=True,
        prominence_scale=4.5,
        min_distance=20,
        min_width=3,
        top_k=3,
        smooth_window=3,
        drop_localization="center",
    )


def _flights_spike() -> SpikeDetector:
    return SpikeDetector(
        prominence_scale=4.5,
        min_distance=30,
        min_width=1,
        top_k=3,
        smooth_window=1,
        drop_localization="left_edge",
    )


MHC_CHANNEL_CONFIG = ChannelConfig(
    names=CHANNEL_NAMES,
    meta=CHANNEL_META,
    continuous=CONTINUOUS_CHANNELS,
    groups={
        "activity": frozenset(ACTIVITY_CHANNELS),
        "sleep": frozenset(SLEEP_CHANNELS),
    },
    aggregators={"hk_watch:HKQuantityTypeIdentifierHeartRate": NonZeroAggregator()},
    detectors={
        "hk_iphone:HKQuantityTypeIdentifierStepCount":              [TrendDetector(), _activity_spike()],
        "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning": [TrendDetector(), _activity_spike()],
        "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed":         [TrendDetector(), _flights_spike()],
        "hk_watch:HKQuantityTypeIdentifierStepCount":               [TrendDetector(), _activity_spike()],
        "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning":  [TrendDetector(), _activity_spike()],
        "hk_watch:HKQuantityTypeIdentifierHeartRate":               [TrendDetector(filter_zeros=True), _heart_rate_spike()],
        "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned":      [TrendDetector(), _activity_spike()],
    },
)

DATASET_DIR = os.environ.get("MHC_DATASET_DIR", "data/mhc")
