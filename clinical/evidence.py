#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Deterministic findings for structured rationales: what each caption says, which way it
points for a label, and how much weight it carries.

Direction comes from the caption flag (and, for absence signals, from silence), strength
from the table below, so only the weighing prose is left to the LLM and every finding can
be checked against the extractors. Strengths are a draft for clinician review, set from
the interviews and the clinical signal audit (project_plan.tex).
"""
from __future__ import annotations

from dataclasses import dataclass

from timef.schema import Annotation

from clinical.caption_grouper import ABSENCE_SIGNALS, FLAG_SUFFIX, SIGNAL_LABEL_RELEVANCE

# (strength when the signal supports the label, strength when it argues against it).
STRENGTH: dict[str, tuple[str, str]] = {
    "signal_21_circadian_dipping":  ("strong", "moderate"),  # most consistent signal in the audit
    "signal_7_hr_inversion":        ("strong", "weak"),      # C1: "this is weird"; rare, artifact-prone
    "signal_16_hr_spread":          ("moderate", "moderate"),
    "signal_3_resting_hr":          ("moderate", "weak"),
    "signal_20_nocturnal_rhr":      ("moderate", "weak"),
    "signal_22_gait_speed":         ("moderate", "weak"),
    "signal_18_cadence":            ("moderate", "moderate"),  # running cadence is reassuring (C2)
    "signal_12_bradycardia":        ("moderate", "weak"),
    "signal_1_hr_recovery":         ("weak", "weak"),          # confounded by bout timing (audit)
    "signal_17_sustained_bout":     ("weak", "moderate"),      # absence may be lifestyle (C2)
    "signal_24_sedentary_burden":   ("weak", "weak"),          # near-universal in the cohort
    "signal_14_nocturnal_activity": ("weak", "weak"),          # nonspecific (C2)
    "signal_11_peak_hr_proxy":      ("weak", "weak"),
    "signal_6_exercise_volume":     ("weak", "weak"),          # reverse causality (C1)
    "signal_4_sleep_duration":      ("moderate", "weak"),      # AASM/SRS 7-9 h; confounded by lifestyle (C2)
    "signal_4_sleep_regularity":    ("moderate", "weak"),      # C1: erratic timing points to sleep abnormality
    # Demographics: the covariates OpenMHC appends to every feature vector.
    "demographic_age":              ("strong", "moderate"),    # set per band in demographic_findings
    "demographic_bmi":              ("moderate", "weak"),
}

DEMOGRAPHIC_LABELS = {
    "demographic_age": {"cardiovascular_disease", "Hypertension", "CAD", "Heart Failure or CHF",
                        "Diabetes", "Atrial fibrillation (Afib)", "Cerebrovascular Disease",
                        "sleep_diagnosis1", "PH"},
    "demographic_bmi": {"cardiovascular_disease", "Hypertension", "Diabetes", "Heart Failure or CHF",
                        "sleep_diagnosis1", "CAD", "PH", "Cerebrovascular Disease",
                        "Atrial fibrillation (Afib)"},
}
STRENGTH_ORDER = {"strong": 0, "moderate": 1, "weak": 2}


@dataclass(frozen=True)
class Finding:
    signal: str
    text: str
    supports: bool
    strength: str

    def render(self, label_display: str) -> str:
        direction = "supports" if self.supports else "argues against"
        return f"{self.text.rstrip('.')} → {direction} {label_display}, {self.strength}"


def demographic_findings(age: float | None, bmi: float | None, label: str) -> list[Finding]:
    """Age and BMI as findings. Bands: age >= 60 strong support, 50-59 moderate, 40-49 weak,
    < 40 moderate against; BMI >= 30 moderate support, 25-30 weak, < 25 weak against."""
    out: list[Finding] = []
    if age is not None and age > 0 and label in DEMOGRAPHIC_LABELS["demographic_age"]:
        if age >= 60:
            supports, strength = True, "strong"
        elif age >= 50:
            supports, strength = True, "moderate"
        elif age >= 40:
            supports, strength = True, "weak"
        else:
            supports, strength = False, "moderate"
        out.append(Finding("demographic_age", f"Age {age:.0f} years", supports, strength))
    if bmi is not None and bmi > 0 and label in DEMOGRAPHIC_LABELS["demographic_bmi"]:
        if bmi >= 30:
            supports, strength, band = True, "moderate", "obese"
        elif bmi >= 25:
            supports, strength, band = True, "weak", "overweight"
        else:
            supports, strength, band = False, "weak", "normal or below"
        out.append(Finding("demographic_bmi", f"BMI {bmi:.1f} ({band})", supports, strength))
    return out


def findings_for(signal_results: dict[str, list[Annotation]], label: str) -> list[Finding]:
    """Findings relevant to ``label``, strongest first (supporting before against on ties)."""
    out: list[Finding] = []
    for signal, annotations in signal_results.items():
        if label not in SIGNAL_LABEL_RELEVANCE.get(signal, []):
            continue
        support_strength, against_strength = STRENGTH[signal]
        if signal in ABSENCE_SIGNALS and not annotations:
            out.append(Finding(signal, ABSENCE_SIGNALS[signal], True, support_strength))
            continue
        for ann in annotations:
            supports = str(ann.label or "").endswith(FLAG_SUFFIX)
            out.append(Finding(signal, ann.text, supports,
                               support_strength if supports else against_strength))
    return sorted(out, key=lambda f: (STRENGTH_ORDER[f.strength], not f.supports))
