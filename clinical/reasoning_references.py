#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Scientific reference text for each clinical signal, for inclusion in LLM prompts.

These are the "How to reason from it" blocks from project_plan.tex, extracted here
so the rationale prompt is built from auditable, prompt-controlled references (the LLM
is *given* this information, never asked to recall it). Keys match the signal names in
clinical.caption_grouper.SIGNAL_LABEL_RELEVANCE.

Per the project plan: lab-derived cutoffs (HRR, chronotropic %) are measured in
controlled conditions and don't transfer directly to free-living watch data, so they
are framed as rough anchors, not hard thresholds. Each block carries its own caveat.
"""
from __future__ import annotations

REASONING_REFERENCES: dict[str, str] = {
    "signal_1_hr_recovery": (
        "HR recovery after exercise. Clinically, a drop of <=12 bpm in the first minute "
        "after a graded exercise test carries ~4x higher mortality risk (Cole et al., NEJM "
        "1999). But that cutoff assumes a controlled lab test with a precise stop moment; "
        "free-living workouts do not have one, so do not apply the bpm number. Reason "
        "directionally instead: a longer return to baseline is worse, a shorter return is "
        "healthier. Compare within this person, not to a fixed cutoff."
    ),
    "signal_3_resting_hr": (
        "Resting (overnight) heart rate. Awake resting HR: normal range 60-100 bpm, "
        "considered healthy below 80; risk rises with higher rates. A meta-analysis found "
        "each +10 bpm in resting HR raises all-cause mortality by ~9% and cardiovascular "
        "mortality by ~8% (Zhang et al., CMAJ 2016). Caveat: this value is measured during "
        "sleep, which runs a few bpm lower than seated resting HR, so do not apply the "
        "awake cutoffs directly. Treat higher as worse, but do not call a hard line."
    ),
    "signal_20_nocturnal_rhr": (
        "Overnight resting-HR trend (infection / illness signal). A sustained rise of "
        ">=3-4 bpm above the person's normal baseline across several consecutive nights is "
        "an early sign of infection, typically 2-3 days before symptoms appear (NightSignal; "
        "Alavi et al., Nat. Med. 2022). About 3 bpm is the median fluctuation over 3 months "
        "for a healthy person. Caveat: the baseline here is only the preceding ~7 nights "
        "(short), so a per-night rising trend is more robust than a single deviation number."
    ),
    "signal_12_bradycardia": (
        "Profound nocturnal bradycardia. Very low HR during sleep is associated with sleep "
        "apnea: during apneic events the heart rate can drop sharply. It occurs in ~7-40% of "
        "sleep apnea patients and is greatly reduced by CPAP treatment (72-89% reduction). "
        "Caveat: HR in the 30s-40s during deep sleep is normal (strong vagal tone), and "
        "bradycardia alone is not enough to diagnose apnea — use it alongside other "
        "supporting signs."
    ),
    "signal_14_nocturnal_activity": (
        "Nocturnal activity (movement inside the sleep window). Nighttime awakenings suggest "
        "sleep fragmentation (nocturia, restlessness, or a late bedtime misidentified as "
        "sleep). The interpretation is highly nonspecific. One-off events mean little; a "
        "pattern recurring across nights is more meaningful. Note it and hedge."
    ),
    "signal_16_hr_spread": (
        "Peak-to-minimum HR spread (max HR minus min HR; this is not HRV). A wide spread "
        "(very low resting minimum and very high exercise peak) reflects a healthy heart "
        "with good dynamic range. A narrow spread (a high resting minimum together with a "
        "low exercise peak, both compressed toward the middle) is unfavorable: the heart "
        "can neither go low at rest nor high under load. The combination of both ends being "
        "compressed is more informative than either end alone."
    ),
    "signal_17_sustained_bout": (
        "Sustained activity duration. The inability to sustain continuous moderate activity "
        "(>=10 consecutive minutes at >=100 steps/min, the Tudor-Locke moderate-intensity "
        "threshold) is a sign of poor vascular reserve or exertional intolerance. The signal "
        "is the longest single bout, not the average — short bursts of normal daily movement "
        "do not count. Caveat: a short longest-bout might mean the person cannot sustain "
        "activity (pathological) or simply did not that week (sedentary by choice) — hedge "
        "heavily."
    ),
    "signal_18_cadence": (
        "Step cadence (peak steps/min). Reaching vigorous-intensity cadence (~130 steps/min, "
        "~6 METs; Tudor-Locke) is reassuring and argues for good cardiovascular reserve. "
        "Never exceeding a low cadence "
        "(~30 steps/min) could suggest neurological or vascular impairment (e.g. a shuffling "
        "gait). Direction: reaching a high cadence argues against disease; an inability to "
        "reach it argues for it. Caveat: never reaching a high cadence may be a lifestyle "
        "choice, not an inability."
    ),
    "signal_21_circadian_dipping": (
        "Circadian HR dipping (nocturnal vs. daytime HR). In healthy people HR drops 10-20% "
        "overnight relative to daytime resting levels. Non-dipping of heart rate (a night/day "
        "HR ratio >0.90, i.e. <10% drop) independently predicts cardiovascular events: ~2.4x "
        "higher CV risk in hypertensive patients, independent of blood-pressure non-dipping "
        "(Eguchi et al., 2009), and it predicts fatal and nonfatal CV events in the general "
        "population too (Cuspidi et al., J. Hypertens. 2018). Reverse dipping (overnight HR "
        "actually higher than daytime) "
        "is a stronger warning sign. Caveat: poor sleep alignment and noisy wake-time "
        "measurement can produce the same pattern in healthy people, so weigh the evidence "
        "rather than applying a hard verdict."
    ),
    "signal_22_gait_speed": (
        "Gait speed (distance per unit time during walking bouts). Comfortable gait speed "
        "below 0.8 m/s indicates functional limitation; below 1.0 m/s predicts adverse "
        "outcomes (Studenski et al., JAMA 2011). A short stride at moderate cadence "
        "(shuffling) can suggest peripheral vascular disease, deconditioning, or neurological "
        "impairment. Use as a gradient: slower is worse; do not apply a hard cutoff."
    ),
    "signal_24_sedentary_burden": (
        "Sedentary burden (longest unbroken inactivity during waking hours). Prolonged, "
        "uninterrupted sedentary bouts independently predict all-cause mortality, beyond "
        "total sedentary volume alone (Diaz et al., Ann. Int. Med. 2017; total sedentary "
        "time also predicts mortality per Biswas et al., Ann. Int. Med. 2015). The common "
        "recommendation is to break sitting at least every 30 minutes. Flag very long "
        "unbroken runs; hedge for desk workers and people with mobility limitations."
    ),
    "signal_6_exercise_volume": (
        "Weekly exercise volume. Guideline: >=150 min/week of moderate activity, or >=75 "
        "min/week of vigorous, plus muscle-strengthening on >=2 days/week (AHA; WHO 2020; "
        "PAG 2018). Below guideline = insufficient activity. Caveat: sick or disabled people "
        "also move less, so low activity predicts disease but disease also causes low "
        "activity. Treat as a healthy-habit indicator, not a diagnosis."
    ),
    "signal_11_peak_hr_proxy": (
        "Peak HR as a fitness proxy (weekly maximum HR vs. age-predicted max, 220 - age). "
        "Chronotropic incompetence is failing to reach ~80-85% of age-predicted max HR during "
        "hard exercise (thresholds vary 70-85%; Lauer et al., JAMA 1999 / Circulation 1996). "
        "Reaching >=85% during a hard "
        "bout is normal; <80% is blunted. Caveat: in free-living data a low peak might just "
        "mean the person did not push hard that week — only treat it as informative when "
        "there is clear evidence of genuine high effort (e.g. high kcal/min) alongside the "
        "low peak."
    ),
    "signal_4_sleep_duration": (
        "Sleep duration. Adults need at least 7 and under 9 hours per night (AASM/SRS consensus, "
        "Watson et al., J. Clin. Sleep Med. 2015); habitual short or long sleep is common in "
        "insomnia and sleep apnea. Caveat: at hourly resolution durations can be overestimated by "
        "up to an hour, and lifestyle, shift work and alcohol produce the same pattern."
    ),
    "signal_4_sleep_regularity": (
        "Sleep regularity. Erratic sleep onset from night to night points toward a sleep "
        "abnormality (clinician interview); irregular sleep timing is associated with poorer sleep "
        "health (Phillips et al., Sci. Rep. 2017). Caveat: unconventional schedules and weekends "
        "produce the same pattern, so note it and hedge."
    ),
    "demographic_age": (
        "Age. Cardiovascular risk and hypertension prevalence rise steeply with age: in US "
        "adults hypertension affects roughly a quarter of people aged 18-39, about 55% at "
        "40-59 and about 75% at 60 or older (CDC NCHS Data Brief 364, 2020). Age is the single "
        "strongest risk factor in the cohort, but it does not by itself establish disease."
    ),
    "demographic_bmi": (
        "Body mass index. Obesity (BMI >= 30) roughly doubles to triples the risk of "
        "hypertension and raises cardiovascular and diabetes risk; overweight (25-30) carries a "
        "smaller increase (WHO; AHA scientific statement on obesity and cardiovascular disease, "
        "Circulation 2021). Treat it as a risk factor, not a diagnosis."
    ),
    "signal_7_hr_inversion": (
        "HR inversion (HR higher during sleep than during exercise on the same day). Rare; "
        "in free-living watch recordings it is usually caused by sensor dropout (watch "
        "slipping, poor contact) rather than real physiology. When genuine, the clinical "
        "hypothesis is a rate-limiting medication (e.g. a beta-blocker) capping the exercise "
        "HR. We have no medication data, so note the inversion but do not diagnose its cause."
    ),
}


def references_for(signal_names: list[str]) -> list[tuple[str, str]]:
    """Return (signal_name, reference_text) pairs for the given fired signals.

    Signals without a reference entry are skipped. Order follows ``signal_names``.
    """
    return [
        (name, REASONING_REFERENCES[name])
        for name in signal_names
        if name in REASONING_REFERENCES
    ]
