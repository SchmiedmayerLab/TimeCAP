#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import pathlib

import numpy as np

from detectors.bundle import BundleDetectionResult, BundleDetector
from extractors import CaptionExtractor
from timef.schema import Annotation, Recording

TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent.parent / "templates" / "templates.json"
BUNDLE_TEMPLATES: dict[str, list[str]] = json.loads(TEMPLATES_PATH.read_text())["bundle"]


class BundleExtractor(CaptionExtractor):
    caption_type = "bundle"

    def __init__(self, config):
        super().__init__(config)
        self.detector = BundleDetector(config)

    def extract(self, row: Recording) -> list[Annotation]:
        seed = self._seed(row.row_id)
        annotations: list[Annotation] = []
        for offset, result in enumerate(self.detector.detect(row)):
            templates = BUNDLE_TEMPLATES[result.bundle_type]
            template = templates[(seed + offset) % len(templates)]
            template_vars = self._format_template_vars(result)
            annotations.append(
                Annotation(
                    caption_type=self.caption_type,
                    text=template.format(**template_vars),
                    channel_idxs=result.channel_idxs,
                    window=result.window,
                    label=result.label,
                )
            )
        return annotations

    def _format_template_vars(self, result: BundleDetectionResult) -> dict[str, str]:
        template_vars = dict(result.template_vars)
        if result.bundle_type in {"awake_in_bed", "restful_sleep"}:
            hr_mean = float(template_vars.pop("hr_mean"))
            template_vars["hr_phrase"] = _hr_phrase(hr_mean)
            template_vars["hr_sentence"] = _hr_sentence(hr_mean)
        elif result.bundle_type in {"stair_climb_burst", "stairs_effort"}:
            hr_mean = float(template_vars.pop("hr_mean"))
            template_vars["hr_phrase"] = _hr_phrase(hr_mean)
            template_vars["flights_total"] = _format_value(float(template_vars["flights_total"]), 0)
            if "steps_total" in template_vars:
                template_vars["steps_total"] = _format_value(float(template_vars["steps_total"]), 0)
        elif result.bundle_type in {"activity_surge", "steady_cardio", "nonstep_cardio"}:
            for key in ("hr_mean", "steps_total", "energy_total", "distance_total"):
                if key in template_vars:
                    decimals = 1 if key == "hr_mean" else 0
                    template_vars[key] = _format_value(float(template_vars[key]), decimals)
            if "distance_total" in result.template_vars:
                template_vars["distance_phrase"] = _distance_phrase(float(result.template_vars["distance_total"]))
            if "energy_total" in result.template_vars:
                template_vars["energy_phrase"] = _energy_phrase(float(result.template_vars["energy_total"]))
        elif result.bundle_type in {"workout_session", "low_motion_workout"}:
            workout_name = str(template_vars.pop("workout_name"))
            hr_mean = float(template_vars.pop("hr_mean"))
            steps_total = float(template_vars.pop("steps_total"))
            distance_total = float(template_vars.pop("distance_total"))
            energy_total = float(template_vars.pop("energy_total"))
            low_motion = bool(template_vars.pop("low_motion"))
            workout_phrase = _workout_phrase(workout_name)
            template_vars["workout_phrase"] = workout_phrase
            template_vars["workout_phrase_cap"] = _capitalize_phrase(workout_phrase)
            template_vars["hr_phrase"] = _hr_phrase(hr_mean)
            template_vars["hr_sentence"] = _hr_sentence(hr_mean)
            template_vars["metric_summary"] = _movement_summary(
                steps_total=steps_total,
                distance_total=distance_total,
                energy_total=energy_total,
                low_motion=low_motion,
            )
        return template_vars


def _format_value(value: float, decimals: int) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def _hr_phrase(value: float) -> str:
    if not np.isfinite(value):
        return "no heart-rate data available"
    return f"a mean heart rate of {value:.1f} bpm"


def _hr_sentence(value: float) -> str:
    if not np.isfinite(value):
        return "heart-rate data was unavailable"
    return f"mean heart rate was {value:.1f} bpm"


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


def _capitalize_phrase(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def _distance_phrase(distance_total: float) -> str:
    if not np.isfinite(distance_total):
        return "n/a distance"
    if distance_total >= 1000:
        return f"{distance_total / 1000.0:.1f} km"
    return f"{distance_total:.0f} m"


def _energy_phrase(energy_total: float) -> str:
    if not np.isfinite(energy_total):
        return "n/a active energy"
    if energy_total >= 1000:
        return f"{energy_total / 1000.0:.1f} kcal"
    return f"{energy_total:.0f} cal"


def _movement_summary(
    steps_total: float,
    distance_total: float,
    energy_total: float,
    low_motion: bool = False,
) -> str:
    parts: list[str] = []
    if low_motion:
        if steps_total < 300 and distance_total < 300:
            parts.append("limited movement")
        elif steps_total > 0:
            parts.append(f"{steps_total:.0f} steps")
    else:
        if steps_total >= 50:
            parts.append(f"{steps_total:.0f} steps")
        if distance_total >= 50:
            parts.append(f"{_distance_phrase(distance_total)} traveled")
    parts.append(_energy_phrase(energy_total))
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{parts[0]}, {parts[1]}, and {parts[2]}"
