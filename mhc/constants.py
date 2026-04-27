#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
import os

from extractors import ChannelConfig
from aggregators import DurationAggregator, NonZeroAggregator, TotalAggregator
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
WATCH_HR_CHANNEL = "hk_watch:HKQuantityTypeIdentifierHeartRate"
WATCH_DISTANCE_CHANNEL = "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning"
WATCH_STEP_CHANNEL = "hk_watch:HKQuantityTypeIdentifierStepCount"
WATCH_ACTIVE_ENERGY_CHANNEL = "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned"
ELEVATED_HEART_RATE_BPM = 100.0

CARDIO_WORKOUT_CHANNELS = (
    (
        "workout:HKWorkoutActivityTypeRunning",
        "cardio_running",
        "running",
    ),
    (
        "workout:HKWorkoutActivityTypeCycling",
        "cardio_cycling",
        "cycling",
    ),
)

STATIONARY_WORKOUT_CHANNELS = (
    (
        "workout:HKWorkoutActivityTypeHighIntensityIntervalTraining",
        "stationary_hiit",
        "stationary HIIT",
    ),
    (
        "workout:HKWorkoutActivityTypeTraditionalStrengthTraining",
        "stationary_strength",
        "traditional strength",
    ),
    (
        "workout:HKWorkoutActivityTypeFunctionalStrengthTraining",
        "stationary_functional",
        "functional",
    ),
)


MHC_CHANNEL_CONFIG = ChannelConfig(
    names=CHANNEL_NAMES,
    meta=CHANNEL_META,
    continuous=CONTINUOUS_CHANNELS,
    groups={
        "activity": frozenset(ACTIVITY_CHANNELS),
        "sleep": frozenset(SLEEP_CHANNELS),
    },
    aggregators={
        "hk_iphone:HKQuantityTypeIdentifierStepCount": TotalAggregator(),
        "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning": TotalAggregator(),
        "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed": TotalAggregator(),
        "hk_watch:HKQuantityTypeIdentifierStepCount": TotalAggregator(),
        "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning": TotalAggregator(),
        "hk_watch:HKQuantityTypeIdentifierHeartRate": NonZeroAggregator(),
        **{channel: DurationAggregator() for channel in ACTIVITY_CHANNELS},
        **{channel: DurationAggregator() for channel in SLEEP_CHANNELS},
    },
    detectors={
        "hk_iphone:HKQuantityTypeIdentifierStepCount":              [TrendDetector(), SpikeDetector(min_height=100.0, min_distance=10, min_width=1, top_k=4)],
        "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning": [TrendDetector(), SpikeDetector(min_height=72.0, min_distance=10, min_width=1, top_k=4)],
        "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed":         [TrendDetector(), SpikeDetector(min_height=1.0, min_distance=10, min_width=1, top_k=4)],
        "hk_watch:HKQuantityTypeIdentifierStepCount":               [TrendDetector(), SpikeDetector(min_height=100.0, min_distance=10, min_width=1, top_k=4)],
        "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning":  [TrendDetector(), SpikeDetector(min_height=72.0, min_distance=10, min_width=1, top_k=4)],
        "hk_watch:HKQuantityTypeIdentifierHeartRate":               [TrendDetector(filter_zeros=True), SpikeDetector(filter_zeros=True, min_height=100.0, min_prominence=5.0, min_distance=20, min_width=1, top_k=4)],
        "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned":      [TrendDetector(), SpikeDetector(min_height=3000.0, min_distance=15, min_width=1, top_k=4)],
    },
)

DATASET_DIR = os.environ.get("MHC_DATASET_DIR", "data/mhc")
