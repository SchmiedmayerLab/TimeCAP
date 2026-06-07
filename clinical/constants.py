#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Channel name constants used by clinical synthesizers.

Mirrors mhc/constants.py but avoids the mhc package-level import chain
(mhc/__init__.py → mhc.dataset → torch).
"""

WATCH_HR_CHANNEL = "hk_watch:HKQuantityTypeIdentifierHeartRate"
WATCH_STEP_CHANNEL = "hk_watch:HKQuantityTypeIdentifierStepCount"
WATCH_DISTANCE_CHANNEL = "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning"
WATCH_ACTIVE_ENERGY_CHANNEL = "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned"

SLEEP_ASLEEP_CHANNEL = "sleep:asleep"
SLEEP_INBED_CHANNEL = "sleep:inbed"

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
