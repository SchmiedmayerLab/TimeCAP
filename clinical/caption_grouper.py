#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Group clinical signal annotations into per-label indicating / not-indicating buckets.

Usage::

    from clinical.caption_grouper import group_captions, SIGNAL_LABEL_RELEVANCE
    from clinical.auditor import FLAG_SUFFIX

    signal_results = {
        "signal_3_resting_hr": synthesizer_3.synthesize(rec, config),
        "signal_17_sustained_bout": synthesizer_17.synthesize(rec, config),
        ...
    }
    grouped = group_captions(signal_results)
    # grouped["cardiovascular_disease"].indicating   → list of concerning caption texts
    # grouped["cardiovascular_disease"].not_indicating → list of reassuring caption texts
"""
from __future__ import annotations

from dataclasses import dataclass, field

from timef.schema import Annotation

FLAG_SUFFIX = ":flag"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIGNAL_LABEL_RELEVANCE: dict[str, list[str]] = {
    "signal_3_resting_hr": [
        "cardiovascular_disease", "CAD", "Hypertension", "Diabetes", "sleep_diagnosis1",
    ],
    "signal_1_hr_recovery": [
        "cardiovascular_disease", "CAD", "Heart Failure or CHF",
    ],
    "signal_17_sustained_bout": [
        "cardiovascular_disease", "CAD", "Heart Failure or CHF",
        "Diabetes", "Cerebrovascular Disease",
    ],
    "signal_11_peak_hr_proxy": [
        "cardiovascular_disease", "CAD", "Heart Failure or CHF",
    ],
    "signal_24_sedentary_burden": [
        "cardiovascular_disease", "CAD", "Heart Failure or CHF",
        "Diabetes", "Hypertension", "Cerebrovascular Disease",
    ],
    "signal_6_exercise_volume": [
        "cardiovascular_disease", "CAD", "Heart Failure or CHF",
        "Diabetes", "Cerebrovascular Disease",
    ],
    "signal_7_hr_inversion": [
        "cardiovascular_disease", "Atrial fibrillation (Afib)",
    ],
    "signal_21_circadian_dipping": [
        "cardiovascular_disease", "Hypertension",
        "Atrial fibrillation (Afib)", "sleep_diagnosis1", "Heart Failure or CHF",
    ],
    "signal_16_hr_spread": [
        "cardiovascular_disease", "CAD", "Heart Failure or CHF",
        "Hypertension", "Diabetes",
    ],
    "signal_20_nocturnal_rhr": [
        "Hypertension", "sleep_diagnosis1", "Atrial fibrillation (Afib)",
    ],
    "signal_14_nocturnal_activity": [
        "sleep_diagnosis1", "Heart Failure or CHF",
    ],
    "signal_22_gait_speed": [
        "cardiovascular_disease", "Cerebrovascular Disease", "Heart Failure or CHF",
    ],
    "signal_18_cadence": [
        "cardiovascular_disease", "Cerebrovascular Disease", "Diabetes",
    ],
}

# Signals where *absence* of firing is the disease indicator.
# When these never fire, a synthetic indicating caption is generated.
ABSENCE_SIGNALS: dict[str, str] = {
    "signal_17_sustained_bout": (
        "No sustained activity bouts (≥10 consecutive minutes at ≥100 steps/min) "
        "were detected — the participant did not achieve a continuous moderate-intensity "
        "effort during this recording."
    ),
}


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class LabelCaptions:
    """Grouped captions for a single clinical label."""
    label: str
    indicating: list[str] = field(default_factory=list)
    not_indicating: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.indicating and not self.not_indicating

    def __repr__(self) -> str:
        return (
            f"LabelCaptions({self.label!r}, "
            f"indicating={len(self.indicating)}, "
            f"not_indicating={len(self.not_indicating)})"
        )


# ---------------------------------------------------------------------------
# Core grouper
# ---------------------------------------------------------------------------

def group_captions(
    signal_results: dict[str, list[Annotation]],
    labels: list[str] | None = None,
    relevance: dict[str, list[str]] = SIGNAL_LABEL_RELEVANCE,
) -> dict[str, LabelCaptions]:
    """Group annotations into per-label indicating / not-indicating buckets.

    Args:
        signal_results: Mapping from signal name to the list of Annotation objects
            returned by its synthesizer. Pass an empty list for signals that did
            not fire — absence signals are handled automatically.
        labels: Subset of labels to compute. Defaults to all labels in ``relevance``.
        relevance: Signal-to-label relevance map. Defaults to SIGNAL_LABEL_RELEVANCE.

    Returns:
        Dict mapping label name → LabelCaptions with .indicating and .not_indicating
        lists of caption texts.
    """
    target_labels: set[str] = set(labels) if labels is not None else {
        lbl for lbls in relevance.values() for lbl in lbls
    }

    grouped: dict[str, LabelCaptions] = {lbl: LabelCaptions(label=lbl) for lbl in target_labels}

    for signal_name, annotations in signal_results.items():
        relevant_labels = [lbl for lbl in relevance.get(signal_name, []) if lbl in target_labels]
        if not relevant_labels:
            continue

        is_absence_signal = signal_name in ABSENCE_SIGNALS

        if is_absence_signal and not annotations:
            # Absence of firing is the disease indicator
            absence_text = ABSENCE_SIGNALS[signal_name]
            for lbl in relevant_labels:
                grouped[lbl].indicating.append(absence_text)
        else:
            for ann in annotations:
                text = ann.text
                is_flag = str(ann.label or "").endswith(FLAG_SUFFIX)
                for lbl in relevant_labels:
                    if is_flag:
                        grouped[lbl].indicating.append(text)
                    else:
                        grouped[lbl].not_indicating.append(text)

    return grouped


def format_grouped_captions(grouped: dict[str, LabelCaptions]) -> str:
    """Render grouped captions as a structured prompt block for an LLM.

    Produces one section per label that has at least one caption, formatted as::

        === cardiovascular_disease ===
        Indicating:
          - Resting HR during sleep: mean 83 bpm ...
        Not indicating:
          - Sustained 15-min activity bout at 124 steps/min ...

    Labels with no captions are omitted.
    """
    parts: list[str] = []
    for lbl, caps in grouped.items():
        if caps.is_empty():
            continue
        lines = [f"=== {lbl} ==="]
        if caps.indicating:
            lines.append("Indicating:")
            lines.extend(f"  - {t}" for t in caps.indicating)
        if caps.not_indicating:
            lines.append("Not indicating:")
            lines.extend(f"  - {t}" for t in caps.not_indicating)
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
