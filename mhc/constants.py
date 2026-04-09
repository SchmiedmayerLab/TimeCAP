#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
import os
from typing import Any

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


CHANNEL_SPIKE_PARAMS: dict[str, dict[str, Any]] = {
    # Activity channels use absolute minima so thresholds can be justified in domain units.
    "hk_iphone:HKQuantityTypeIdentifierStepCount": {
        "min_height": 100.0,
        "min_distance": 10,
        "min_width": 1,
        "top_k": 4,
    },
    "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning": {
        "min_height": 72.0,
        "min_distance": 10,
        "min_width": 1,
        "top_k": 4,
    },
    "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed": {
        "min_height": 1.0,
        "min_distance": 10,
        "min_width": 1,
        "top_k": 4,
    },
    "hk_watch:HKQuantityTypeIdentifierStepCount": {
        "min_height": 100.0,
        "min_distance": 10,
        "min_width": 1,
        "top_k": 4,
    },
    "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning": {
        "min_height": 72.0,
        "min_distance": 10,
        "min_width": 1,
        "top_k": 4,
    },
    "hk_watch:HKQuantityTypeIdentifierHeartRate": {
        "filter_zeros": True,
        "min_height": 100.0,
        "min_prominence": 5.0,
        "min_distance": 20,
        "min_width": 1,
        "top_k": 4,
    },
    "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned": {
        "min_height": 3000,
        "min_distance": 15,
        "min_width": 1,
        "top_k": 4,
    },
}


def channel_spike_detector(channel_name: str, **overrides: Any) -> SpikeDetector:
    params = dict(CHANNEL_SPIKE_PARAMS.get(channel_name, {}))
    params.update(overrides)
    return SpikeDetector(**params)


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
        "hk_iphone:HKQuantityTypeIdentifierStepCount":              [TrendDetector(), channel_spike_detector("hk_iphone:HKQuantityTypeIdentifierStepCount")],
        "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning": [TrendDetector(), channel_spike_detector("hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning")],
        "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed":         [TrendDetector(), channel_spike_detector("hk_iphone:HKQuantityTypeIdentifierFlightsClimbed")],
        "hk_watch:HKQuantityTypeIdentifierStepCount":               [TrendDetector(), channel_spike_detector("hk_watch:HKQuantityTypeIdentifierStepCount")],
        "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning":  [TrendDetector(), channel_spike_detector("hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning")],
        "hk_watch:HKQuantityTypeIdentifierHeartRate":               [TrendDetector(filter_zeros=True), channel_spike_detector("hk_watch:HKQuantityTypeIdentifierHeartRate")],
        "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned":      [TrendDetector(), channel_spike_detector("hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned")],
    },
)

DATASET_DIR = os.environ.get("MHC_DATASET_DIR", "data/mhc")
