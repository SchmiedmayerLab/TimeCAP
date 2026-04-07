#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
import os

from extractors import ChannelConfig
from aggregators import NonZeroAggregator
from detectors.drop import DropDetector
from detectors.nonwear import NonWearDetector
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
        "min_distance": 15,
        "min_width": 1,
        "top_k": 4,
    }
    params.update(overrides)
    return SpikeDetector(**params)


def _activity_drop(**overrides) -> DropDetector:
    params = {
        "min_distance": 15,
        "min_width": 1,
        "top_k": 4,
        "drop_localization": "center",
        "classify_return_to_baseline": True,
        "baseline_percentile": 0.20,
        "baseline_abs_tolerance": 1.0,
        "baseline_rel_tolerance": 0.25,
        "reversion_margin": 1.0,
        "persistence_window": 2,
    }
    params.update(overrides)
    return DropDetector(**params)


def _heart_rate_spike() -> SpikeDetector:
    return SpikeDetector(
        filter_zeros=True,
        min_distance=20,
        min_width=1,
        top_k=4,
    )


def _heart_rate_drop() -> DropDetector:
    return DropDetector(
        filter_zeros=True,
        min_distance=20,
        min_width=1,
        top_k=4,
        drop_localization="center",
        classify_return_to_baseline=False,
        baseline_percentile=0.15,
        baseline_abs_tolerance=4.0,
        baseline_rel_tolerance=0.10,
        reversion_margin=4.0,
        persistence_window=2,
    )


def _flights_spike() -> SpikeDetector:
    return SpikeDetector(
        min_distance=30,
        min_width=1,
        top_k=4,
    )


def _flights_drop() -> DropDetector:
    return DropDetector(
        min_distance=30,
        min_width=1,
        top_k=4,
        drop_localization="center",
        classify_return_to_baseline=True,
        baseline_percentile=0.20,
        baseline_abs_tolerance=0.5,
        baseline_rel_tolerance=0.25,
        reversion_margin=0.5,
        persistence_window=2,
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
        "hk_iphone:HKQuantityTypeIdentifierStepCount":              [TrendDetector(), _activity_spike(), _activity_drop()],
        "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning": [TrendDetector(), _activity_spike(), _activity_drop()],
        "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed":         [TrendDetector(), _flights_spike(), _flights_drop()],
        "hk_watch:HKQuantityTypeIdentifierStepCount":               [NonWearDetector(), TrendDetector(), _activity_spike(), _activity_drop()],
        "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning":  [NonWearDetector(), TrendDetector(), _activity_spike(), _activity_drop()],
        "hk_watch:HKQuantityTypeIdentifierHeartRate":               [NonWearDetector(), TrendDetector(filter_zeros=True), _heart_rate_spike(), _heart_rate_drop()],
        "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned":      [NonWearDetector(), TrendDetector(), _activity_spike(), _activity_drop()],
    },
)

DATASET_DIR = os.environ.get("MHC_DATASET_DIR", "data/mhc")
