#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""One finding per clinical signal for a week of hourly data (19 x 168).

Each day is spread to minute resolution (hourly sums / 60, rates repeated) so the daily
synthesizers run unchanged; their per-day values and flags are then summarized into a
single weekly finding. A signal supports the label when it flags on at least half of its
measurable days. Signals that do not survive hourly resolution are left out: cadence
(hourly means always look low), HR recovery, bradycardia, sustained bout and gait speed
(minute-level transients), weekly peak HR (its threshold flags almost every week at
hourly resolution) and nocturnal RHR (duplicates resting HR).
"""
from __future__ import annotations

import numpy as np

from clinical.circadian_dipping import CircadianDippingSynthesizer
from clinical.data import CHANNEL_NAMES, DAILY_CLINICAL_CONFIG, WEEKLY_CLINICAL_CONFIG
from clinical.evidence import STRENGTH, STRENGTH_ORDER, Finding
from clinical.caption_grouper import FLAG_SUFFIX, SIGNAL_LABEL_RELEVANCE
from clinical.exercise_volume import ExerciseVolumeSynthesizer
from clinical.hr_inversion import HRInversionSynthesizer
from clinical.hr_spread import HRSpreadSynthesizer
from clinical.nocturnal_activity import NocturnalActivitySynthesizer
from clinical.resting_hr import RestingHRSynthesizer
from clinical.sedentary_burden import SedentaryBurdenSynthesizer
from timef.schema import Recording

# Hourly channels that are sums over the hour (steps, distance, flights); spread evenly.
HOURLY_SUM_IDXS = [0, 1, 2, 3, 4]

DAILY = {
    "signal_3_resting_hr": RestingHRSynthesizer(),
    "signal_21_circadian_dipping": CircadianDippingSynthesizer(),
    "signal_16_hr_spread": HRSpreadSynthesizer(),
    "signal_7_hr_inversion": HRInversionSynthesizer(),
    "signal_24_sedentary_burden": SedentaryBurdenSynthesizer(),
    "signal_14_nocturnal_activity": NocturnalActivitySynthesizer(),
}
# Signals that only caption when something is found: a measured day without a caption
# counts as "not flagged" rather than "not evaluated".
SILENT_WHEN_NORMAL = {"signal_7_hr_inversion", "signal_24_sedentary_burden",
                      "signal_14_nocturnal_activity"}
EXERCISE = ExerciseVolumeSynthesizer()


def _recording(row_id: str, user_id: str, date: str, values: np.ndarray) -> Recording:
    return Recording(
        row_id=row_id, user_id=user_id, date=date, values=values,
        channel_names=tuple(CHANNEL_NAMES), display_names=tuple(CHANNEL_NAMES),
        units=tuple([""] * len(CHANNEL_NAMES)),
        has_any_data=np.isfinite(values).any(axis=1),
        channel_variance=np.zeros(len(CHANNEL_NAMES), dtype=np.float32),
    )


def day_recordings(week: Recording) -> list[Recording]:
    """Seven minute-resolution day Recordings from a weekly hourly Recording."""
    days = []
    for d in range(7):
        hourly = np.asarray(week.values[:, d * 24:(d + 1) * 24], dtype=np.float32).copy()
        hourly[HOURLY_SUM_IDXS] /= 60.0
        days.append(_recording(f"{week.row_id}:d{d}", week.user_id, week.date,
                               np.repeat(hourly, 60, axis=1)))
    return days


def _per_day(signal: str, days: list[Recording]) -> list[tuple[dict, bool]]:
    """(features, flagged) for every measurable day."""
    synth = DAILY[signal]
    out = []
    for rec in days:
        values = synth.features(rec)
        if not any(np.isfinite(v) for v in values.values()):
            continue
        annotations = synth.synthesize(rec, DAILY_CLINICAL_CONFIG)
        if not annotations and signal not in SILENT_WHEN_NORMAL:
            continue
        out.append((values, any(str(a.label or "").endswith(FLAG_SUFFIX) for a in annotations)))
    return out


def _median(rows: list[tuple[dict, bool]], key: str) -> float:
    return float(np.nanmedian([r[0][key] for r in rows]))


def _text(signal: str, rows: list[tuple[dict, bool]]) -> str:
    n, k = len(rows), sum(flagged for _, flagged in rows)
    if signal == "signal_3_resting_hr":
        return (f"Resting HR during sleep: median {_median(rows, 'mean_hr'):.0f} bpm across "
                f"{n} nights; above 80 bpm on {k} of {n}")
    if signal == "signal_21_circadian_dipping":
        return (f"Circadian HR dip (daytime resting vs. overnight): median "
                f"{_median(rows, 'dip_pct'):.1f}%; non-dipping (<10%) on {k} of {n} days")
    if signal == "signal_16_hr_spread":
        return (f"Daily HR range (hourly averages): median {_median(rows, 'min_hr'):.0f}–"
                f"{_median(rows, 'max_hr'):.0f} bpm; compressed (min ≥60 and max ≤120 bpm) "
                f"on {k} of {n} days")
    if signal == "signal_7_hr_inversion":
        return (f"Sleep HR above exercise HR on {k} of {n} days with both measured "
                f"(median sleep minus exercise {_median(rows, 'sleep_minus_exercise'):+.0f} bpm)")
    if signal == "signal_24_sedentary_burden":
        return (f"Longest unbroken sedentary period during waking hours: median "
                f"{_median(rows, 'longest_min'):.0f} min; at least one ≥60-min sedentary bout "
                f"on {k} of {n} days")
    if signal == "signal_14_nocturnal_activity":
        return f"Movement during the sleep window on {k} of {n} nights"
    raise KeyError(signal)


SLEEP_LABELS = {"sleep_diagnosis1"}


def sleep_episodes(week: Recording, min_hours: int = 3) -> list[tuple[int, int]]:
    """Main sleep episodes as (start hour, length) from the hourly sleep:asleep channel.
    An hour counts as asleep if any minute was; one-hour gaps inside an episode are bridged."""
    asleep = np.asarray(week.values[CHANNEL_NAMES.index("sleep:asleep")], dtype=float)
    on = np.isfinite(asleep) & (asleep > 0)
    for t in range(1, len(on) - 1):
        if not on[t] and on[t - 1] and on[t + 1]:
            on[t] = True
    episodes, start = [], None
    for t, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = t
        elif not v and start is not None:
            if t - start >= min_hours:
                episodes.append((start, t - start))
            start = None
    return episodes


def sleep_findings(week: Recording) -> list[Finding]:
    """Sleep duration against the 7-9 h guideline and night-to-night onset regularity."""
    episodes = sleep_episodes(week)
    out: list[Finding] = []
    if len(episodes) >= 2:
        hours = [length for _, length in episodes]
        outside = sum(1 for h in hours if h < 7 or h > 9)
        supports = outside * 2 >= len(hours)
        out.append(Finding(
            "signal_4_sleep_duration",
            f"Sleep duration (hourly resolution): median {np.median(hours):.0f} h across {len(hours)} "
            f"nights; outside 7–9 h on {outside} of {len(hours)}",
            supports, STRENGTH["signal_4_sleep_duration"][0 if supports else 1]))
    if len(episodes) >= 3:
        # Onset as hours after noon, so 23:00 and 01:00 are 11 and 13, not 23 and 1.
        onsets = [((start % 24) - 12) % 24 for start, _ in episodes]
        sd = float(np.std(onsets))
        supports = sd > 1.0
        out.append(Finding(
            "signal_4_sleep_regularity",
            f"Sleep onset varies by {sd:.1f} h (SD) across {len(onsets)} nights; "
            f"{'irregular (>1 h)' if supports else 'regular (≤1 h)'}",
            supports, STRENGTH["signal_4_sleep_regularity"][0 if supports else 1]))
    return out


def weekly_findings(week: Recording, label: str) -> list[Finding]:
    """Findings relevant to ``label`` for one weekly window, strongest first."""
    days = day_recordings(week)
    out: list[Finding] = []
    for signal in DAILY:
        if label not in SIGNAL_LABEL_RELEVANCE.get(signal, []):
            continue
        rows = _per_day(signal, days)
        if not rows:
            continue
        supports = sum(flagged for _, flagged in rows) * 2 >= len(rows)
        if signal in SILENT_WHEN_NORMAL and not supports and signal != "signal_24_sedentary_burden":
            # "never happened" findings for rare events add noise; keep only when they recur
            continue
        strength = STRENGTH[signal][0 if supports else 1]
        out.append(Finding(signal, _text(signal, rows), supports, strength))

    if label in SIGNAL_LABEL_RELEVANCE["signal_6_exercise_volume"]:
        minutes = EXERCISE.features(week)["weekly_minutes"]
        if np.isfinite(minutes):
            anns = EXERCISE.synthesize(week, WEEKLY_CLINICAL_CONFIG)
            supports = not anns or any(str(a.label).endswith(FLAG_SUFFIX) for a in anns)
            text = (f"Logged workouts: {minutes:.0f} min across the week (hours with any workout "
                    f"× 60); {'below' if supports else 'meets'} the 150 min/week guideline")
            out.append(Finding("signal_6_exercise_volume", text, supports,
                               STRENGTH["signal_6_exercise_volume"][0 if supports else 1]))
    if label in SLEEP_LABELS:
        out.extend(sleep_findings(week))
    return sorted(out, key=lambda f: (STRENGTH_ORDER[f.strength], not f.supports))
