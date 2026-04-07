#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from extractors import ChannelConfig
from timef.schema import Recording

WATCH_STEP = "hk_watch:HKQuantityTypeIdentifierStepCount"
IPHONE_STEP = "hk_iphone:HKQuantityTypeIdentifierStepCount"
WATCH_DISTANCE = "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning"
IPHONE_DISTANCE = "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning"
HEART_RATE = "hk_watch:HKQuantityTypeIdentifierHeartRate"
ACTIVE_ENERGY = "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned"
FLIGHTS_CLIMBED = "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed"
SLEEP_ASLEEP = "sleep:asleep"
SLEEP_IN_BED = "sleep:inbed"

LOW_MOTION_WORKOUTS = {
    "workout:HKWorkoutActivityTypeTraditionalStrengthTraining",
    "workout:HKWorkoutActivityTypeFunctionalStrengthTraining",
    "workout:HKWorkoutActivityTypeYoga",
}

SEARCHABLE_BUNDLE_TARGETS = (
    "activity_surge",
    "awake_in_bed",
    "nonstep_cardio",
    "restful_sleep",
    "stair_climb_burst",
    "stairs_effort",
    "steady_cardio",
    "watch_nonwear_context",
)

BUNDLE_PRIORITY = {
    "workout:HKWorkoutActivityTypeHighIntensityIntervalTraining": 120,
    "workout:HKWorkoutActivityTypeRunning": 118,
    "workout:HKWorkoutActivityTypeCycling": 116,
    "workout:HKWorkoutActivityTypeElliptical": 114,
    "workout:HKWorkoutActivityTypeMixedMetabolicCardioTraining": 112,
    "workout:HKWorkoutActivityTypeWalking": 108,
    "workout:HKWorkoutActivityTypeTraditionalStrengthTraining": 106,
    "workout:HKWorkoutActivityTypeFunctionalStrengthTraining": 104,
    "workout:HKWorkoutActivityTypeYoga": 102,
    "workout:HKWorkoutActivityTypeOther": 100,
    "stairs_effort": 92,
    "stair_climb_burst": 90,
    "steady_cardio": 82,
    "nonstep_cardio": 80,
    "watch_nonwear_context": 75,
    "activity_surge": 70,
    "awake_in_bed": 62,
    "restful_sleep": 60,
}


@dataclass(slots=True)
class BundleDetectionResult:
    bundle_type: str
    label: str
    channel_idxs: tuple[int, ...]
    window: tuple[int, int]
    score: float
    template_vars: dict[str, Any] = field(default_factory=dict)


class BundleDetector:
    def __init__(self, config: ChannelConfig):
        self.config = config

    def detect(self, row: Recording) -> list[BundleDetectionResult]:
        scored: list[BundleDetectionResult] = []
        scored.extend(self._sleep_bundle_detections(row))
        scored.extend(self._activity_burst_detections(row))
        scored.extend(self._mobility_bundle_detections(row))
        scored.extend(self._workout_bundle_detections(row))
        scored.extend(self._watch_nonwear_detections(row))
        selected = self._select_non_overlapping_bundles(scored)
        selected.sort(
            key=lambda result: (
                result.window[0],
                -(result.window[1] - result.window[0]),
                -result.score,
            )
        )
        return selected

    def _select_non_overlapping_bundles(
        self,
        scored: list[BundleDetectionResult],
    ) -> list[BundleDetectionResult]:
        if not scored:
            return []
        ranked = sorted(
            scored,
            key=lambda item: (
                -self._bundle_rank(item),
                -item.score,
                -self._window_length(item),
            ),
        )
        selected: list[BundleDetectionResult] = []
        for candidate in ranked:
            if any(self._conflicts(candidate, existing) for existing in selected):
                continue
            selected.append(candidate)
        return selected

    def _bundle_rank(self, result: BundleDetectionResult) -> float:
        priority = BUNDLE_PRIORITY.get(result.label, 50)
        return priority + min(self._window_length(result) / 1000.0, 0.99)

    @staticmethod
    def _window_length(result: BundleDetectionResult) -> int:
        return max(0, int(result.window[1] - result.window[0]))

    def _conflicts(self, left: BundleDetectionResult, right: BundleDetectionResult) -> bool:
        overlap = _window_overlap(left.window, right.window)
        if overlap <= 0:
            return False
        shorter = min(self._window_length(left), self._window_length(right))
        if shorter <= 0:
            return False
        overlap_ratio = overlap / shorter
        if left.label == right.label:
            return overlap_ratio >= 0.5
        sleep_labels = {"awake_in_bed", "restful_sleep"}
        if left.label in sleep_labels and right.label in sleep_labels:
            return overlap_ratio >= 0.8
        return overlap_ratio >= 0.6

    def _sleep_bundle_detections(self, row: Recording) -> list[BundleDetectionResult]:
        idx = {name: i for i, name in enumerate(row.channel_names)}
        required = {SLEEP_IN_BED, SLEEP_ASLEEP, WATCH_STEP, IPHONE_STEP, HEART_RATE}
        if not required.issubset(idx):
            return []
        in_bed = row.values[idx[SLEEP_IN_BED]] > 0
        asleep = row.values[idx[SLEEP_ASLEEP]] > 0
        hr = row.values[idx[HEART_RATE]]
        steps = np.nan_to_num(row.values[idx[WATCH_STEP]], nan=0.0) + np.nan_to_num(row.values[idx[IPHONE_STEP]], nan=0.0)
        hr_valid = hr[np.isfinite(hr)]
        if hr_valid.size == 0:
            return []
        rest_hr_threshold = min(58.0, float(np.percentile(hr_valid, 30)) + 8.0)
        shared_channels = _channel_indices(row, [SLEEP_IN_BED, SLEEP_ASLEEP, HEART_RATE, WATCH_STEP, IPHONE_STEP])
        results: list[BundleDetectionResult] = []
        for bed_start, bed_end in _contiguous_windows(in_bed, 30):
            if not asleep[bed_start:bed_end].any():
                continue
            awake_mask = np.zeros_like(in_bed, dtype=bool)
            awake_mask[bed_start:bed_end] = in_bed[bed_start:bed_end] & ~asleep[bed_start:bed_end]
            for start, end in _contiguous_windows(awake_mask, 15):
                steps_total = float(steps[start:end].sum())
                hr_mean = _safe_mean(hr[start:end])
                if steps_total > 200:
                    continue
                results.append(
                    BundleDetectionResult(
                        bundle_type="awake_in_bed",
                        label="awake_in_bed",
                        channel_idxs=shared_channels,
                        window=(start, end),
                        score=(end - start) + max(0.0, 200.0 - steps_total) / 50.0,
                        template_vars={
                            "start_time": _format_minute(start),
                            "end_time": _format_minute(end),
                            "bed_start_time": _format_minute(bed_start),
                            "bed_end_time": _format_minute(bed_end),
                            "hr_mean": hr_mean,
                        },
                    )
                )
            sleep_mask = np.zeros_like(asleep, dtype=bool)
            sleep_mask[bed_start:bed_end] = asleep[bed_start:bed_end]
            for start, end in _contiguous_windows(sleep_mask, 30):
                steps_total = float(steps[start:end].sum())
                hr_mean = _safe_mean(hr[start:end])
                if not np.isfinite(hr_mean) or steps_total > 20 or hr_mean > rest_hr_threshold:
                    continue
                results.append(
                    BundleDetectionResult(
                        bundle_type="restful_sleep",
                        label="restful_sleep",
                        channel_idxs=shared_channels,
                        window=(start, end),
                        score=(end - start) + max(0.0, rest_hr_threshold - hr_mean),
                        template_vars={
                            "start_time": _format_minute(start),
                            "end_time": _format_minute(end),
                            "hr_mean": hr_mean,
                        },
                    )
                )
        return results

    def _activity_burst_detections(self, row: Recording) -> list[BundleDetectionResult]:
        idx = {name: i for i, name in enumerate(row.channel_names)}
        needed = {WATCH_STEP, IPHONE_STEP, ACTIVE_ENERGY, HEART_RATE, FLIGHTS_CLIMBED}
        if not needed.issubset(idx):
            return []
        steps = np.nan_to_num(row.values[idx[WATCH_STEP]], nan=0.0) + np.nan_to_num(row.values[idx[IPHONE_STEP]], nan=0.0)
        energy = np.nan_to_num(row.values[idx[ACTIVE_ENERGY]], nan=0.0)
        hr = row.values[idx[HEART_RATE]]
        flights = np.nan_to_num(row.values[idx[FLIGHTS_CLIMBED]], nan=0.0)
        hr_valid = hr[np.isfinite(hr)]
        if hr_valid.size == 0:
            return []
        step_threshold = max(3.0, _positive_quantile(steps, 75))
        energy_threshold = max(20.0, _positive_quantile(energy, 75))
        if not np.isfinite(step_threshold) or not np.isfinite(energy_threshold):
            return []
        hr_threshold = max(70.0, float(np.percentile(hr_valid, 70)))
        elevated = ((steps >= step_threshold) | (energy >= energy_threshold)) & np.isfinite(hr) & (hr >= hr_threshold)
        elevated = _dilate_mask(elevated, radius=2)
        channel_names = [WATCH_STEP, IPHONE_STEP, ACTIVE_ENERGY, HEART_RATE]
        if WATCH_DISTANCE in idx:
            channel_names.append(WATCH_DISTANCE)
        if IPHONE_DISTANCE in idx:
            channel_names.append(IPHONE_DISTANCE)
        channel_idxs = _channel_indices(row, channel_names)
        results: list[BundleDetectionResult] = []
        for start, end in _contiguous_windows(elevated, 10):
            steps_total = float(steps[start:end].sum())
            energy_total = float(energy[start:end].sum())
            hr_mean = _safe_mean(hr[start:end])
            flights_total = float(flights[start:end].sum())
            if not np.isfinite(hr_mean) or steps_total < 200 or energy_total < 5000:
                continue
            if flights_total >= 3 and steps_total <= 2500:
                continue
            results.append(
                BundleDetectionResult(
                    bundle_type="activity_surge",
                    label="activity_surge",
                    channel_idxs=channel_idxs,
                    window=(start, end),
                    score=steps_total / 200.0 + energy_total / 20000.0 + hr_mean,
                    template_vars={
                        "start_time": _format_minute(start),
                        "end_time": _format_minute(end),
                        "hr_mean": hr_mean,
                        "steps_total": steps_total,
                        "energy_total": energy_total,
                    },
                )
            )
        return results

    def _mobility_bundle_detections(self, row: Recording) -> list[BundleDetectionResult]:
        idx = {name: i for i, name in enumerate(row.channel_names)}
        needed = {WATCH_STEP, IPHONE_STEP, WATCH_DISTANCE, IPHONE_DISTANCE, ACTIVE_ENERGY, HEART_RATE, FLIGHTS_CLIMBED}
        if not needed.issubset(idx):
            return []
        steps = np.nan_to_num(row.values[idx[WATCH_STEP]], nan=0.0) + np.nan_to_num(row.values[idx[IPHONE_STEP]], nan=0.0)
        distance = np.nan_to_num(row.values[idx[WATCH_DISTANCE]], nan=0.0) + np.nan_to_num(row.values[idx[IPHONE_DISTANCE]], nan=0.0)
        energy = np.nan_to_num(row.values[idx[ACTIVE_ENERGY]], nan=0.0)
        hr = row.values[idx[HEART_RATE]]
        flights = np.nan_to_num(row.values[idx[FLIGHTS_CLIMBED]], nan=0.0)
        results: list[BundleDetectionResult] = []
        stair_channels = _channel_indices(row, [FLIGHTS_CLIMBED, WATCH_STEP, IPHONE_STEP, HEART_RATE, ACTIVE_ENERGY, WATCH_DISTANCE, IPHONE_DISTANCE])
        stair_mask = _dilate_mask((flights > 0) | (steps >= 15), radius=1)
        for start, end in _contiguous_windows(stair_mask, 2):
            flights_total = float(flights[start:end].sum())
            steps_total = float(steps[start:end].sum())
            energy_total = float(energy[start:end].sum())
            hr_mean = _safe_mean(hr[start:end])
            if flights_total < 3 or steps_total < 20:
                continue
            score = flights_total * 10.0 + steps_total / 40.0 + max(0.0, energy_total / 20000.0)
            if np.isfinite(hr_mean):
                score += hr_mean
            results.append(BundleDetectionResult(
                bundle_type="stair_climb_burst",
                label="stair_climb_burst",
                channel_idxs=stair_channels,
                window=(start, end),
                score=score,
                template_vars={
                    "start_time": _format_minute(start),
                    "end_time": _format_minute(end),
                    "flights_total": flights_total,
                    "steps_total": steps_total,
                    "hr_mean": hr_mean,
                },
            ))
        stairs_effort_channels = _channel_indices(row, [FLIGHTS_CLIMBED, HEART_RATE, WATCH_STEP, IPHONE_STEP, ACTIVE_ENERGY])
        stairs_effort_mask = _dilate_mask((flights >= 1) & np.isfinite(hr) & (hr >= 75), radius=1)
        for start, end in _contiguous_windows(stairs_effort_mask, 3):
            flights_total = float(flights[start:end].sum())
            hr_mean = _safe_mean(hr[start:end])
            steps_total = float(steps[start:end].sum())
            if flights_total < 2 or not np.isfinite(hr_mean):
                continue
            results.append(BundleDetectionResult(
                bundle_type="stairs_effort",
                label="stairs_effort",
                channel_idxs=stairs_effort_channels,
                window=(start, end),
                score=flights_total * 12.0 + hr_mean + min(steps_total, 1000.0) / 100.0,
                template_vars={
                    "start_time": _format_minute(start),
                    "end_time": _format_minute(end),
                    "flights_total": flights_total,
                    "hr_mean": hr_mean,
                },
            ))
        cardio_channels = _channel_indices(row, [WATCH_STEP, IPHONE_STEP, WATCH_DISTANCE, IPHONE_DISTANCE, HEART_RATE, ACTIVE_ENERGY])
        cardio_mask = _dilate_mask((steps >= 10) & (distance >= 8) & np.isfinite(hr) & (hr >= 80), radius=2)
        for start, end in _contiguous_windows(cardio_mask, 15):
            steps_total = float(steps[start:end].sum())
            distance_total = float(distance[start:end].sum())
            energy_total = float(energy[start:end].sum())
            hr_mean = _safe_mean(hr[start:end])
            if steps_total < 600 or distance_total < 400 or not np.isfinite(hr_mean):
                continue
            results.append(BundleDetectionResult(
                bundle_type="steady_cardio",
                label="steady_cardio",
                channel_idxs=cardio_channels,
                window=(start, end),
                score=hr_mean + steps_total / 150.0 + distance_total / 200.0 + energy_total / 30000.0,
                template_vars={
                    "start_time": _format_minute(start),
                    "end_time": _format_minute(end),
                    "hr_mean": hr_mean,
                    "steps_total": steps_total,
                    "distance_total": distance_total,
                    "energy_total": energy_total,
                },
            ))
        nonstep_channels = _channel_indices(row, [HEART_RATE, ACTIVE_ENERGY, WATCH_STEP, IPHONE_STEP, WATCH_DISTANCE, IPHONE_DISTANCE])
        nonstep_mask = _dilate_mask((energy >= 50) & np.isfinite(hr) & (hr >= 85) & (steps <= 5), radius=2)
        for start, end in _contiguous_windows(nonstep_mask, 15):
            steps_total = float(steps[start:end].sum())
            energy_total = float(energy[start:end].sum())
            hr_mean = _safe_mean(hr[start:end])
            if energy_total < 6000 or not np.isfinite(hr_mean):
                continue
            results.append(BundleDetectionResult(
                bundle_type="nonstep_cardio",
                label="nonstep_cardio",
                channel_idxs=nonstep_channels,
                window=(start, end),
                score=hr_mean + energy_total / 15000.0 + max(0.0, 50.0 - steps_total),
                template_vars={
                    "start_time": _format_minute(start),
                    "end_time": _format_minute(end),
                    "hr_mean": hr_mean,
                    "steps_total": steps_total,
                    "energy_total": energy_total,
                },
            ))
        return results

    def _workout_bundle_detections(self, row: Recording) -> list[BundleDetectionResult]:
        idx = {name: i for i, name in enumerate(row.channel_names)}
        workout_names = [name for name in row.channel_names if name.startswith("workout:")]
        if not workout_names:
            return []
        hr = row.values[idx[HEART_RATE]] if HEART_RATE in idx else np.full(row.values.shape[1], np.nan, dtype=float)
        watch_steps = np.nan_to_num(row.values[idx[WATCH_STEP]], nan=0.0) if WATCH_STEP in idx else np.zeros(row.values.shape[1], dtype=float)
        iphone_steps = np.nan_to_num(row.values[idx[IPHONE_STEP]], nan=0.0) if IPHONE_STEP in idx else np.zeros(row.values.shape[1], dtype=float)
        watch_dist = np.nan_to_num(row.values[idx[WATCH_DISTANCE]], nan=0.0) if WATCH_DISTANCE in idx else np.zeros(row.values.shape[1], dtype=float)
        iphone_dist = np.nan_to_num(row.values[idx[IPHONE_DISTANCE]], nan=0.0) if IPHONE_DISTANCE in idx else np.zeros(row.values.shape[1], dtype=float)
        energy = np.nan_to_num(row.values[idx[ACTIVE_ENERGY]], nan=0.0) if ACTIVE_ENERGY in idx else np.zeros(row.values.shape[1], dtype=float)
        results: list[BundleDetectionResult] = []
        for workout_name in workout_names:
            workout_mask = row.values[idx[workout_name]] > 0
            for start, end in _contiguous_windows(workout_mask, 10):
                steps_total = float((watch_steps[start:end] + iphone_steps[start:end]).sum())
                distance_total = float((watch_dist[start:end] + iphone_dist[start:end]).sum())
                energy_total = float(energy[start:end].sum())
                hr_mean = _safe_mean(hr[start:end])
                if (not np.isfinite(hr_mean) and steps_total < 50 and distance_total < 50 and energy_total < 5000):
                    continue
                bundle_type = "low_motion_workout" if workout_name in LOW_MOTION_WORKOUTS else "workout_session"
                results.append(BundleDetectionResult(
                    bundle_type=bundle_type,
                    label=workout_name,
                    channel_idxs=_channel_indices(row, [workout_name, HEART_RATE, WATCH_STEP, IPHONE_STEP, WATCH_DISTANCE, IPHONE_DISTANCE, ACTIVE_ENERGY]),
                    window=(start, end),
                    score=(end - start) + energy_total / 20000.0 + steps_total / 100.0 + (hr_mean if np.isfinite(hr_mean) else 0.0),
                    template_vars={
                        "start_time": _format_minute(start),
                        "end_time": _format_minute(end),
                        "workout_name": workout_name,
                        "hr_mean": hr_mean,
                        "steps_total": steps_total,
                        "distance_total": distance_total,
                        "energy_total": energy_total,
                        "low_motion": bundle_type == "low_motion_workout",
                    },
                ))
        return results

    def _watch_nonwear_detections(self, row: Recording) -> list[BundleDetectionResult]:
        idx = {name: i for i, name in enumerate(row.channel_names)}
        needed = {WATCH_STEP, WATCH_DISTANCE, HEART_RATE, ACTIVE_ENERGY, IPHONE_STEP, IPHONE_DISTANCE, SLEEP_ASLEEP, SLEEP_IN_BED}
        if not needed.issubset(idx):
            return []
        watch_step = row.values[idx[WATCH_STEP]]
        watch_distance = row.values[idx[WATCH_DISTANCE]]
        heart_rate = row.values[idx[HEART_RATE]]
        watch_energy = row.values[idx[ACTIVE_ENERGY]]
        phone_steps = np.nan_to_num(row.values[idx[IPHONE_STEP]], nan=0.0)
        phone_distance = np.nan_to_num(row.values[idx[IPHONE_DISTANCE]], nan=0.0)
        asleep = row.values[idx[SLEEP_ASLEEP]] > 0
        in_bed = row.values[idx[SLEEP_IN_BED]] > 0
        workout_masks = {name: (row.values[idx[name]] > 0) for name in row.channel_names if name.startswith("workout:")}
        watch_missing = np.isnan(heart_rate)
        watch_missing &= np.nan_to_num(watch_step, nan=0.0) <= 0
        watch_missing &= np.nan_to_num(watch_distance, nan=0.0) <= 0
        watch_missing &= np.nan_to_num(watch_energy, nan=0.0) <= 0
        context_mask = (phone_steps > 0) | (phone_distance > 0) | asleep | in_bed
        for mask in workout_masks.values():
            context_mask |= mask
        windows = _contiguous_windows(watch_missing & context_mask, 20)
        if not windows:
            return []
        channel_idxs = _channel_indices(row, [WATCH_STEP, WATCH_DISTANCE, HEART_RATE, ACTIVE_ENERGY, IPHONE_STEP, IPHONE_DISTANCE, SLEEP_ASLEEP, SLEEP_IN_BED, *workout_masks.keys()])
        results: list[BundleDetectionResult] = []
        for start, end in windows:
            context_phrase = _watch_nonwear_context_phrase(start, end, phone_steps, phone_distance, asleep, in_bed, workout_masks)
            if context_phrase is None:
                continue
            score = (end - start) + (10 if ("sleep" in context_phrase or "bed" in context_phrase) else 0) + (6 if "phone" in context_phrase else 0)
            results.append(BundleDetectionResult(
                bundle_type="watch_nonwear_context",
                label="watch_nonwear_context",
                channel_idxs=channel_idxs,
                window=(start, end),
                score=score,
                template_vars={
                    "start_time": _format_minute(start),
                    "end_time": _format_minute(end),
                    "context_phrase": context_phrase,
                },
            ))
        return results


def _contiguous_windows(mask: np.ndarray, min_minutes: int) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end - start >= min_minutes]


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = np.ones(radius * 2 + 1, dtype=int)
    return np.convolve(mask.astype(np.int8), kernel, mode="same") > 0


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmean(values))


def _positive_quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return float("nan")
    return float(np.percentile(positive, q))


def _format_minute(minute: int) -> str:
    minute = int(minute) % 1440
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _workout_phrase(workout_name: str) -> str:
    phrases = {
        "workout:HKWorkoutActivityTypeWalking": "a walking session",
        "workout:HKWorkoutActivityTypeCycling": "a cycling session",
        "workout:HKWorkoutActivityTypeRunning": "a running session",
        "workout:HKWorkoutActivityTypeOther": "an uncategorized workout",
        "workout:HKWorkoutActivityTypeMixedMetabolicCardioTraining": "a mixed cardio workout",
        "workout:HKWorkoutActivityTypeTraditionalStrengthTraining": "a traditional strength workout",
        "workout:HKWorkoutActivityTypeElliptical": "an elliptical workout",
        "workout:HKWorkoutActivityTypeHighIntensityIntervalTraining": "a HIIT workout",
        "workout:HKWorkoutActivityTypeFunctionalStrengthTraining": "a functional strength workout",
        "workout:HKWorkoutActivityTypeYoga": "a yoga session",
    }
    return phrases.get(workout_name, "a workout")


def _watch_nonwear_context_phrase(
    start: int,
    end: int,
    phone_steps: np.ndarray,
    phone_distance: np.ndarray,
    asleep: np.ndarray,
    in_bed: np.ndarray,
    workout_masks: dict[str, np.ndarray],
) -> str | None:
    window = slice(start, end)
    phrases: list[str] = []
    if asleep[window].any():
        phrases.append("sleep was recorded")
    elif in_bed[window].any():
        phrases.append("in-bed time was recorded")
    workout_totals = []
    for workout_name, mask in workout_masks.items():
        minutes = int(np.count_nonzero(mask[window]))
        if minutes > 0:
            workout_totals.append((minutes, _workout_phrase(workout_name)))
    if workout_totals:
        workout_totals.sort(reverse=True)
        phrases.append(f"{workout_totals[0][1]} was recorded")
    phone_step_total = float(phone_steps[window].sum())
    phone_distance_total = float(phone_distance[window].sum())
    if phone_step_total >= 50 or phone_distance_total >= 50:
        if phone_step_total >= 250 or phone_distance_total >= 200:
            phrases.append("the phone continued recording walking activity")
        else:
            phrases.append("the phone continued recording activity")
    if not phrases:
        return None
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{phrases[0]}, {phrases[1]}, and {phrases[2]}"


def _channel_indices(row: Recording, names: list[str]) -> tuple[int, ...]:
    index_by_name = {name: i for i, name in enumerate(row.channel_names)}
    return tuple(dict.fromkeys(index_by_name[name] for name in names if name in index_by_name))


def _window_overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))
