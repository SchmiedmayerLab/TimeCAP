#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Generate LaTeX report with signal plots + captions for clinical demonstration cases.

Covers all disease labels available in the 20-user export.
Run from repo root:
    python generate_cvd_report.py

Outputs:
    cvd_report/<label>_<case>.pdf  (signal plots)
    cvd_report/report.tex
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pyarrow.feather as feather

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ARROW_FILE = "clinical_captions_20users/recordings_0000.arrow"
LABELS_FILE = "weekly_labels.parquet"
OUT_DIR = Path("cvd_report")

DISEASE_LABELS = [
    "cardiovascular_disease",
    "CAD",
    "Hypertension",
    "Diabetes",
    "sleep_diagnosis1",
    "Heart Failure or CHF",
    "Atrial fibrillation (Afib)",
    "Cerebrovascular Disease",
]

LABEL_DISPLAY = {
    "cardiovascular_disease":    "cardiovascular disease",
    "CAD":                       "coronary artery disease (CAD)",
    "Hypertension":              "hypertension",
    "Diabetes":                  "diabetes",
    "sleep_diagnosis1":          "sleep disorder",
    "Heart Failure or CHF":      "heart failure (CHF)",
    "Atrial fibrillation (Afib)":"atrial fibrillation",
    "Cerebrovascular Disease":   "cerebrovascular disease",
}

LABEL_LATEX = {
    "cardiovascular_disease":    "Cardiovascular Disease",
    "CAD":                       "Coronary Artery Disease (CAD)",
    "Hypertension":              "Hypertension",
    "Diabetes":                  "Diabetes",
    "sleep_diagnosis1":          "Sleep Disorder",
    "Heart Failure or CHF":      "Heart Failure (CHF)",
    "Atrial fibrillation (Afib)":"Atrial Fibrillation",
    "Cerebrovascular Disease":   "Cerebrovascular Disease",
}

ANN_TO_SIGNAL: dict[str, str] = {
    "resting_hr":         "signal_3_resting_hr",
    "hr_recovery":        "signal_1_hr_recovery",
    "sustained_bout":     "signal_17_sustained_bout",
    "hr_spread":          "signal_16_hr_spread",
    "circadian_dipping":  "signal_21_circadian_dipping",
    "nocturnal_rhr":      "signal_20_nocturnal_rhr",
    "nocturnal_activity": "signal_14_nocturnal_activity",
    "gait_speed":         "signal_22_gait_speed",
    "cadence":            "signal_18_cadence",
    "sedentary_burden":   "signal_24_sedentary_burden",
    "hr_inversion":       "signal_7_hr_inversion",
}

SIGNAL_LABEL_RELEVANCE: dict[str, list[str]] = {
    "signal_3_resting_hr":      ["cardiovascular_disease", "CAD", "Hypertension", "Diabetes", "sleep_diagnosis1"],
    "signal_1_hr_recovery":     ["cardiovascular_disease", "CAD", "Heart Failure or CHF"],
    "signal_17_sustained_bout": ["cardiovascular_disease", "CAD", "Heart Failure or CHF", "Diabetes", "Cerebrovascular Disease"],
    "signal_11_peak_hr_proxy":  ["cardiovascular_disease", "CAD", "Heart Failure or CHF"],
    "signal_24_sedentary_burden": ["cardiovascular_disease", "CAD", "Heart Failure or CHF", "Diabetes", "Hypertension", "Cerebrovascular Disease"],
    "signal_6_exercise_volume": ["cardiovascular_disease", "CAD", "Heart Failure or CHF", "Diabetes", "Cerebrovascular Disease"],
    "signal_7_hr_inversion":    ["cardiovascular_disease", "Atrial fibrillation (Afib)"],
    "signal_21_circadian_dipping": ["cardiovascular_disease", "Hypertension", "Atrial fibrillation (Afib)", "sleep_diagnosis1", "Heart Failure or CHF"],
    "signal_16_hr_spread":      ["cardiovascular_disease", "CAD", "Heart Failure or CHF", "Hypertension", "Diabetes"],
    "signal_20_nocturnal_rhr":  ["Hypertension", "sleep_diagnosis1", "Atrial fibrillation (Afib)"],
    "signal_14_nocturnal_activity": ["sleep_diagnosis1", "Heart Failure or CHF"],
    "signal_22_gait_speed":     ["cardiovascular_disease", "Cerebrovascular Disease", "Heart Failure or CHF"],
    "signal_18_cadence":        ["cardiovascular_disease", "Cerebrovascular Disease", "Diabetes"],
}

# Precompute: disease label -> set of relevant annotation signal types
LABEL_RELEVANT_ANNS: dict[str, set[str]] = {}
for _ann_type, _sig_key in ANN_TO_SIGNAL.items():
    for _lbl in SIGNAL_LABEL_RELEVANCE.get(_sig_key, []):
        LABEL_RELEVANT_ANNS.setdefault(_lbl, set()).add(_ann_type)

# Annotation types that require sleep tracking to be present
SLEEP_DEPENDENT_ANNS = {"resting_hr", "nocturnal_rhr", "nocturnal_activity"}

CH_HR    = "hk_watch:HKQuantityTypeIdentifierHeartRate"
CH_STEPS = "hk_watch:HKQuantityTypeIdentifierStepCount"
CH_SLEEP = "sleep:asleep"

# ---------------------------------------------------------------------------
# Signal scoring
# ---------------------------------------------------------------------------

def score_row(anns: list[dict], relevant_sig_types: set[str]) -> tuple[int, int]:
    """(n_unique_flag_types, n_unique_nonflag_types) for relevant signals only."""
    seen_flag, seen_nonflag = set(), set()
    for a in anns:
        label = str(a.get("label") or "")
        sig_type = label.removesuffix(":flag")
        if sig_type not in relevant_sig_types:
            continue
        if label.endswith(":flag"):
            seen_flag.add(sig_type)
        else:
            seen_nonflag.add(sig_type)
    return len(seen_flag), len(seen_nonflag)


def get_relevant_captions(anns: list[dict], relevant_sig_types: set[str]) -> tuple[list[str], list[str]]:
    """Deduplicated indicating / not-indicating captions for relevant signals."""
    seen_flag: dict[str, str] = {}
    seen_nonflag: dict[str, str] = {}
    for a in anns:
        label = str(a.get("label") or "")
        sig_type = label.removesuffix(":flag")
        if sig_type not in relevant_sig_types:
            continue
        text = _escape(a["text"])
        if label.endswith(":flag"):
            if sig_type not in seen_flag:
                seen_flag[sig_type] = text
        else:
            if sig_type not in seen_nonflag:
                seen_nonflag[sig_type] = text
    return list(seen_flag.values()), list(seen_nonflag.values())


def _escape(text: str) -> str:
    return text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")

# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------

def _has_sleep(row: pd.Series) -> bool:
    ch_names = list(row["channel_names"])
    if CH_SLEEP not in ch_names:
        return False
    values = np.array(row["values"])
    return bool(values[ch_names.index(CH_SLEEP)].sum() > 0)


def find_cases(
    df: pd.DataFrame,
    user_label_map: dict[str, int],
    disease_label: str,
) -> dict[str, int | None]:
    relevant = LABEL_RELEVANT_ANNS.get(disease_label, set())
    needs_sleep = bool(relevant & SLEEP_DEPENDENT_ANNS)

    scores = []
    for idx, row in df.iterrows():
        if row.get("wear_pct", 0) < 70:
            continue
        if needs_sleep and not _has_sleep(row):
            continue
        uid = row["user_id"]
        lval = user_label_map.get(uid, -1)
        n_flag, n_nonflag = score_row(row["annotations"], relevant)
        scores.append((idx, uid, lval, n_flag, n_nonflag))

    pos_rows = [(i, f, n) for i, u, l, f, n in scores if l == 1]
    neg_rows = [(i, f, n) for i, u, l, f, n in scores if l == 0]

    # Indicating: positive user, highest flags
    pos_rows_sorted = sorted(pos_rows, key=lambda x: (-x[1], x[2]))
    indicating_idx = pos_rows_sorted[0][0] if pos_rows_sorted else None

    # Not-indicating: negative user, fewest flags, highest non-flags
    # Prefer 0 flags; fall back to 1 flag if no clean candidate exists.
    for max_flags in (0, 1):
        clean_neg = sorted(
            [(i, f, n) for i, f, n in neg_rows if f <= max_flags],
            key=lambda x: (x[1], -x[2]),
        )
        if clean_neg:
            break
    not_ind_idx = clean_neg[0][0] if clean_neg else None

    # Ambiguous: any user with both flags and non-flags, most balanced
    mixed = [(i, u, l, f, n) for i, u, l, f, n in scores if f > 0 and n > 0]
    mixed_sorted = sorted(mixed, key=lambda x: abs(x[3] - x[4]))
    ambig_idx = None
    for i, u, l, f, n in mixed_sorted:
        if i != indicating_idx and i != not_ind_idx:
            ambig_idx = i
            break
    if ambig_idx is None and mixed_sorted:
        ambig_idx = mixed_sorted[0][0]

    return {"indicating": indicating_idx, "not_indicating": not_ind_idx, "ambiguous": ambig_idx}

# ---------------------------------------------------------------------------
# Rationale generation
# ---------------------------------------------------------------------------

def make_rationale(
    indicating: list[str],
    not_indicating: list[str],
    display_label: str,
    case_type: str,
    disease_truth: bool,
) -> str:
    disp = display_label

    def fmt(texts: list[str]) -> str:
        return "; ".join(t.rstrip(".") for t in texts)

    if case_type == "indicating":
        parts = []
        for t in indicating:
            parts.append(t.rstrip(".") + ", indicating elevated risk.")
        if not_indicating:
            parts.append(
                "While " + not_indicating[0].rstrip(".").lower()
                + " falls within the normal range, the converging evidence from"
                + " the flagged signals above outweighs this single reassuring finding."
            )
        parts.append(f"\n\nAnswer: {disp}")
        return " ".join(parts)

    elif case_type == "not_indicating":
        parts = []
        for t in not_indicating:
            parts.append(t.rstrip(".") + ", within healthy reference range.")
        parts.append(
            f"Taken together, no signal exceeds its clinical threshold."
            f" There is no evidence of {disp} in this recording."
        )
        parts.append(f"\n\nAnswer: no {disp}")
        return " ".join(parts)

    elif case_type == "ambiguous":
        if disease_truth:
            # Reassuring signals first, BUT WAIT corrects to positive
            nf_str = fmt(not_indicating) if not_indicating else "some resting metrics appear normal"
            fl_str = fmt(indicating) if indicating else "other signals are flagged"
            return (
                f"{nf_str}; these findings initially appear reassuring and might suggest"
                f" the absence of {disp}."
                f" BUT WAIT, looking at: {fl_str}."
                f" These flagged signals reveal impaired function consistent with {disp}"
                f" even when resting metrics appear normal.\n\nAnswer: {disp}"
            )
        else:
            # Flagged signals first, BUT WAIT corrects to negative
            fl_str = fmt(indicating) if indicating else "some signals appear elevated"
            nf_str = fmt(not_indicating) if not_indicating else "other signals are within range"
            return (
                f"{fl_str}; these findings initially raise concern for {disp}."
                f" BUT WAIT, looking at: {nf_str}."
                f" The overall pattern is more reassuring than it first appears,"
                f" and the flagged signals alone are insufficient to confirm {disp}.\n\nAnswer: no {disp}"
            )

    return ""

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_row(row: dict, out_path: Path) -> None:
    ch_names = list(row["channel_names"])
    values = np.array(row["values"])
    minutes = np.arange(1440)

    def ch(name):
        try:
            return ch_names.index(name)
        except ValueError:
            return None

    i_hr, i_steps, i_sleep = ch(CH_HR), ch(CH_STEPS), ch(CH_SLEEP)

    workout_signal = np.zeros(1440)
    for c in ch_names:
        if c.startswith("workout:"):
            i = ch(c)
            if i is not None:
                workout_signal = np.maximum(workout_signal, values[i])

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1]})
    fig.subplots_adjust(hspace=0.08)

    ax = axes[0]
    if i_hr is not None:
        hr = values[i_hr].copy().astype(float)
        hr[hr == 0] = np.nan
        ax.plot(minutes, hr, color="#c0392b", lw=0.8, alpha=0.9, label="Heart Rate")
        ax.axhline(75, color="#c0392b", lw=0.6, ls="--", alpha=0.4, label="75 bpm threshold")
        ax.legend(fontsize=7, loc="upper right")
    ax.set_ylabel("HR (bpm)", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    _shade_sleep(ax, values[i_sleep] if i_sleep is not None else np.zeros(1440))
    _shade_workout(ax, workout_signal)

    ax = axes[1]
    if i_steps is not None:
        steps = values[i_steps].copy().astype(float)
        steps[steps == 0] = np.nan
        ax.bar(minutes, steps, width=1.0, color="#2980b9", alpha=0.7)
    ax.set_ylabel("Steps/min", fontsize=9)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    _shade_sleep(ax, values[i_sleep] if i_sleep is not None else np.zeros(1440))

    ax = axes[2]
    if i_sleep is not None:
        ax.fill_between(minutes, 0, values[i_sleep], step="mid", color="#6d597a", alpha=0.7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["awake", "asleep"], fontsize=7)
    ax.set_ylabel("Sleep", fontsize=9)
    ax.set_xlabel("Minute of day (0 = midnight)", fontsize=9)
    ax.set_xlim(0, 1440)
    ax.set_xticks([0, 360, 720, 1080, 1440])
    ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"], fontsize=8)

    patches = [
        mpatches.Patch(color="#6d597a", alpha=0.25, label="Sleep window"),
        mpatches.Patch(color="#27ae60", alpha=0.18, label="Workout"),
    ]
    fig.legend(handles=patches, loc="upper left", fontsize=7,
               bbox_to_anchor=(0.01, 0.98), framealpha=0.6)
    fig.suptitle(
        f"User {row['user_id'][:12]}...   Date: {row['date']}", fontsize=10
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _shade_sleep(ax, sleep_signal):
    in_s, start = False, 0
    for m in range(1440):
        if sleep_signal[m] > 0 and not in_s:
            in_s, start = True, m
        elif sleep_signal[m] == 0 and in_s:
            ax.axvspan(start, m, color="#6d597a", alpha=0.12, lw=0)
            in_s = False
    if in_s:
        ax.axvspan(start, 1440, color="#6d597a", alpha=0.12, lw=0)


def _shade_workout(ax, workout_signal):
    in_w, start = False, 0
    for m in range(1440):
        if workout_signal[m] > 0 and not in_w:
            in_w, start = True, m
        elif workout_signal[m] == 0 and in_w:
            ax.axvspan(start, m, color="#27ae60", alpha=0.15, lw=0)
            in_w = False
    if in_w:
        ax.axvspan(start, 1440, color="#27ae60", alpha=0.15, lw=0)

# ---------------------------------------------------------------------------
# LaTeX builder
# ---------------------------------------------------------------------------

CASE_TITLES = {
    "indicating":     "Clearly Indicating",
    "not_indicating": "Clearly Not Indicating",
    "ambiguous":      "Ambiguous",
}

CASE_SUBTITLES = {
    "indicating":     "True positive user (ground truth: positive)",
    "not_indicating": "True negative user (ground truth: negative)",
    "ambiguous":      "Mixed evidence (ground truth noted in Answer)",
}


def build_section(
    disease_label: str,
    case_type: str,
    row: dict,
    disease_truth: bool,
    fig_filename: str,
) -> str:
    relevant = LABEL_RELEVANT_ANNS.get(disease_label, set())
    indicating, not_indicating = get_relevant_captions(row["annotations"], relevant)
    rationale = make_rationale(
        indicating, not_indicating,
        LABEL_DISPLAY[disease_label], case_type, disease_truth,
    )

    def items(lst):
        if not lst:
            return "  \\item \\textit{(none)}"
        return "\n".join(f"  \\item {t}" for t in lst)

    uid = row["user_id"]
    date = row["date"]
    latex_label = LABEL_LATEX[disease_label]
    case_title = CASE_TITLES[case_type]
    case_sub = CASE_SUBTITLES[case_type]

    rationale_latex = rationale.replace("\n\n", "\n\n\\noindent ")

    return textwrap.dedent(f"""\
        \\subsection*{{{case_title}: {latex_label}}}
        \\textit{{{case_sub}}}\\quad
        \\texttt{{User: {uid[:14]}...}}\\quad Date: {date}

        \\begin{{figure}}[h]
          \\centering
          \\includegraphics[width=\\textwidth]{{{fig_filename}}}
        \\end{{figure}}

        \\textbf{{Indicating {latex_label.lower()}:}}
        \\begin{{itemize}}
        {items(indicating)}
        \\end{{itemize}}

        \\textbf{{Not indicating {latex_label.lower()}:}}
        \\begin{{itemize}}
        {items(not_indicating)}
        \\end{{itemize}}

        \\textbf{{Rationale:}}\\par
        {rationale_latex}

        \\vspace{{1em}}\\hrule\\vspace{{1em}}
        """)


def build_latex(sections: list[str]) -> str:
    body = "\n".join(sections)
    return textwrap.dedent(f"""\
        \\documentclass[11pt,a4paper]{{article}}
        \\usepackage[margin=2.5cm]{{geometry}}
        \\usepackage{{graphicx}}
        \\usepackage{{parskip}}
        \\usepackage{{microtype}}
        \\title{{Clinical Signal Demonstrations\\\\\\large Wearable Sensor Rationale Examples}}
        \\author{{SensorTSLM Clinical Pipeline}}
        \\date{{}}
        \\begin{{document}}
        \\maketitle

        \\noindent
        Each example shows one day of wearable sensor data for a single user.
        Signals are grouped into those \\textbf{{indicating}} the disease label
        (exceeding clinical thresholds) and those \\textbf{{not indicating}} concern.
        The rationale demonstrates the expected chain-of-thought reasoning,
        ending with \\textbf{{Answer: [label]}}.

        \\bigskip\\hrule\\bigskip

        {body}
        \\end{{document}}
        """)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = feather.read_table(ARROW_FILE).to_pandas()

    df_lab = pd.read_parquet(LABELS_FILE)
    arrow_users = set(df["user_id"].unique())
    user_labels_df = (
        df_lab[df_lab["user_id"].isin(arrow_users)]
        .groupby("user_id")[DISEASE_LABELS]
        .max()
        .reset_index()
    )

    OUT_DIR.mkdir(exist_ok=True)
    plot_cache: dict[int, Path] = {}
    sections: list[str] = []

    for disease_label in DISEASE_LABELS:
        user_label_map = dict(zip(
            user_labels_df["user_id"],
            user_labels_df[disease_label],
        ))

        n_pos = sum(1 for v in user_label_map.values() if v == 1)
        print(f"\n=== {disease_label} ({n_pos} positive users) ===")

        if n_pos == 0:
            print("  No positive users -- skipping indicating/ambiguous cases.")
            case_types = ["not_indicating"]
        else:
            case_types = ["indicating", "not_indicating", "ambiguous"]

        cases = find_cases(df, user_label_map, disease_label)

        for case_type in case_types:
            row_idx = cases.get(case_type)
            if row_idx is None:
                print(f"  [{case_type}] no suitable recording found, skipping.")
                continue

            row = df.iloc[row_idx].to_dict()
            uid = row["user_id"]
            disease_truth = user_label_map.get(uid, -1) == 1

            # Plot (cached by row index)
            if row_idx not in plot_cache:
                safe_label = disease_label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
                fig_path = OUT_DIR / f"{safe_label}_{case_type}.pdf"
                print(f"  [{case_type}] row {row_idx}, user {uid[:16]}..., truth={disease_truth} -> {fig_path.name}")
                plot_row(row, fig_path)
                plot_cache[row_idx] = fig_path
            else:
                fig_path = plot_cache[row_idx]
                print(f"  [{case_type}] row {row_idx} (plot reused: {fig_path.name})")

            section = build_section(
                disease_label, case_type, row, disease_truth, fig_path.name
            )
            sections.append(section)

    tex = build_latex(sections)
    tex_path = OUT_DIR / "report.tex"
    tex_path.write_text(tex)
    print(f"\nWrote {tex_path} ({len(sections)} sections)")
    print(f"To compile: cd {OUT_DIR} && pdflatex report.tex")


if __name__ == "__main__":
    main()
