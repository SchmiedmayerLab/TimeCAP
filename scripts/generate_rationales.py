#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Generate per-label clinical rationales from the caption pipeline via an LLM.

For each recording, run the relevant clinical synthesizers, group their captions into
indicating / not-indicating buckets for one target label, build a prompt that pairs the
captions with the scientific reference for each fired signal, and ask an OpenAI-compatible
model to weigh the evidence and produce a rationale ending in a verdict line.

The model call is text-only (no plots): we pass ``signals=[]`` to ClientModel.process,
so the LLM reasons purely over the caption text and the provided references.

Example::

    OPENAI_API_KEY=... MHC_DATASET_DIR=/path/to/daily \\
    python scripts/generate_rationales.py \\
        --variant daily --label cardiovascular_disease \\
        --labels_file weekly_labels.parquet \\
        --model gpt-4o --max_users 20 \\
        --out rationales/cvd.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical.caption_grouper import (
    ABSENCE_SIGNALS,
    SIGNAL_LABEL_RELEVANCE,
    group_captions,
)
from clinical.evidence import findings_for
from clinical.reasoning_references import references_for
from models.client import ClientConfig, ClientModel

# ── Synthesizer registries (mirror scripts/audit_clinical_signals.py) ────────

def _daily_synthesizers() -> dict:
    from clinical.bradycardia import NocturnalBradycardiaSynthesizer
    from clinical.cadence import CadenceSynthesizer
    from clinical.circadian_dipping import CircadianDippingSynthesizer
    from clinical.gait_speed import GaitSpeedSynthesizer
    from clinical.hr_inversion import HRInversionSynthesizer
    from clinical.hr_recovery import HRRecoverySynthesizer
    from clinical.hr_spread import HRSpreadSynthesizer
    from clinical.nocturnal_activity import NocturnalActivitySynthesizer
    from clinical.nocturnal_rhr import NocturnalRHRSynthesizer
    from clinical.resting_hr import RestingHRSynthesizer
    from clinical.sedentary_burden import SedentaryBurdenSynthesizer
    from clinical.sustained_bout import SustainedBoutSynthesizer

    return {
        "signal_3_resting_hr":          RestingHRSynthesizer(),
        "signal_20_nocturnal_rhr":      NocturnalRHRSynthesizer(),
        "signal_12_bradycardia":        NocturnalBradycardiaSynthesizer(),
        "signal_14_nocturnal_activity": NocturnalActivitySynthesizer(),
        "signal_16_hr_spread":          HRSpreadSynthesizer(),
        "signal_17_sustained_bout":     SustainedBoutSynthesizer(),
        "signal_18_cadence":            CadenceSynthesizer(),
        "signal_21_circadian_dipping":  CircadianDippingSynthesizer(),
        "signal_22_gait_speed":         GaitSpeedSynthesizer(),
        "signal_24_sedentary_burden":   SedentaryBurdenSynthesizer(),
        "signal_1_hr_recovery":         HRRecoverySynthesizer(),
        "signal_7_hr_inversion":        HRInversionSynthesizer(),
    }


def _weekly_synthesizers() -> dict:
    from clinical.exercise_volume import ExerciseVolumeSynthesizer
    from clinical.peak_hr_proxy import PeakHRProxySynthesizer

    return {
        "signal_6_exercise_volume": ExerciseVolumeSynthesizer(),
        "signal_11_peak_hr_proxy":  PeakHRProxySynthesizer(),
    }


# Signal-name sets per variant, derived from the synthesizer registries without
# instantiating (the arrow path doesn't need the synthesizer objects).
DAILY_SIGNAL_NAMES: tuple[str, ...] = (
    "signal_3_resting_hr", "signal_20_nocturnal_rhr", "signal_12_bradycardia",
    "signal_14_nocturnal_activity", "signal_16_hr_spread", "signal_17_sustained_bout",
    "signal_18_cadence", "signal_21_circadian_dipping", "signal_22_gait_speed",
    "signal_24_sedentary_burden", "signal_1_hr_recovery", "signal_7_hr_inversion",
)
WEEKLY_SIGNAL_NAMES: tuple[str, ...] = (
    "signal_6_exercise_volume", "signal_11_peak_hr_proxy",
)

# Map a synthesizer's emitted annotation label (with any ":flag" suffix stripped)
# back to its signal name. Most signal names are "signal_<N>_<base>", so the base is
# the suffix after the number — except bradycardia, whose label diverges.
BASE_LABEL_TO_SIGNAL: dict[str, str] = {
    name.split("_", 2)[2]: name for name in SIGNAL_LABEL_RELEVANCE
}
BASE_LABEL_TO_SIGNAL["nocturnal_bradycardia"] = "signal_12_bradycardia"


# Friendlier disease names for the prompt / verdict line. Falls back to the raw
# column name with underscores replaced.
DISPLAY_LABELS: dict[str, str] = {
    "cardiovascular_disease": "cardiovascular disease",
    "CAD": "coronary artery disease (CAD)",
    "Heart Failure or CHF": "heart failure (CHF)",
    "Hypertension": "hypertension",
    "Diabetes": "diabetes",
    "sleep_diagnosis1": "a sleep disorder",
    "Cerebrovascular Disease": "cerebrovascular disease",
    "Atrial fibrillation (Afib)": "atrial fibrillation (Afib)",
}


def _display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label.replace("_", " "))


# ── Prompt construction ──────────────────────────────────────────────────────

# Swappable style instructions. "reconsider" permits the "BUT WAIT" pivot seen in the
# demonstration examples; "plain" asks for a single linear weighing.
STYLE_INSTRUCTIONS: dict[str, str] = {
    "plain": (
        "Weigh the indicating evidence against the not-indicating evidence in a single, "
        "linear explanation. Be explicit about which signals carry more weight and why."
    ),
    "reconsider": (
        "Weigh the indicating evidence against the not-indicating evidence. If the two "
        "sides genuinely conflict, you may reconsider an initial reading mid-way (e.g. "
        "\"BUT WAIT, ...\") before settling on a verdict, the way a careful clinician "
        "revises a first impression. Be explicit about which signals carry more weight."
    ),
}


def _fired_signals(signal_results: dict, label: str) -> list[str]:
    """Signals that contributed a caption for ``label`` (fired, or absence-flagged)."""
    fired: list[str] = []
    for name, anns in signal_results.items():
        if label not in SIGNAL_LABEL_RELEVANCE.get(name, []):
            continue
        if anns or name in ABSENCE_SIGNALS:
            fired.append(name)
    return fired


def _format_buckets(caps) -> str:
    lines: list[str] = []
    if caps.indicating:
        lines.append(f"Indicating {{LABEL}}:")
        lines.extend(f"  - {t}" for t in caps.indicating)
    else:
        lines.append(f"Indicating {{LABEL}}: (none)")
    if caps.not_indicating:
        lines.append(f"Not indicating {{LABEL}}:")
        lines.extend(f"  - {t}" for t in caps.not_indicating)
    else:
        lines.append(f"Not indicating {{LABEL}}: (none)")
    return "\n".join(lines)


def build_prompt(
    label: str,
    caps,
    fired_signals: list[str],
    window_desc: str,
    style: str,
) -> str:
    disp = _display_label(label)
    refs = references_for(fired_signals)
    ref_block = "\n\n".join(f"- {text}" for _, text in refs) or "(none)"
    buckets = _format_buckets(caps).replace("{LABEL}", disp)

    return (
        f"You are a clinical reasoning assistant analyzing wearable-sensor data (Apple "
        f"Watch) from a single participant over {window_desc}. Automatically extracted "
        f"signal captions are grouped below into evidence that does and does not indicate "
        f"{disp}. Decide whether the data, taken together, indicates {disp}.\n\n"
        f"Use only the clinical reference information provided here — do not introduce "
        f"outside facts or numbers. These references are measured mostly in controlled "
        f"settings, so treat them as anchors for free-living watch data, not hard "
        f"thresholds.\n\n"
        f"## Clinical reference information (how to interpret each relevant signal)\n"
        f"{ref_block}\n\n"
        f"## Extracted signals for this recording\n"
        f"{buckets}\n\n"
        f"## Your task\n"
        f"{STYLE_INSTRUCTIONS[style]} Remember that any single signal only gives a hunch, "
        f"not a diagnosis, and that low activity / low effort can reflect lifestyle rather "
        f"than disease. Write the rationale as a few sentences of prose, then end with "
        f"exactly one final line in this form:\n"
        f"Answer: {disp}\n"
        f"or\n"
        f"Answer: no {disp}"
    )


def render_findings(findings, label: str) -> str:
    disp = _display_label(label)
    return "<findings>\n" + "\n".join(f"- {f.render(disp)}" for f in findings) + "\n</findings>"


def build_structured_prompt(label: str, findings, window_desc: str) -> str:
    """Prompt for the structured format: findings (with direction and strength) are fixed
    by the pipeline; the model only writes the weighing and the verdict."""
    disp = _display_label(label)
    refs = references_for(sorted({f.signal for f in findings}))
    ref_block = "\n\n".join(f"- {text}" for _, text in refs) or "(none)"
    return (
        f"You are a clinical reasoning assistant analyzing Apple Watch data from a single "
        f"participant over {window_desc}. The findings below were extracted automatically; "
        f"each states which way it points for {disp} and how much weight it carries. "
        f"Decide whether the data, taken together, indicates {disp}.\n\n"
        f"## Clinical reference information\n{ref_block}\n\n"
        f"## Findings\n{render_findings(findings, label)}\n\n"
        f"## Your task\n"
        f"Write the weighing as 3-5 sentences inside <reasoning></reasoning>, then the verdict "
        f"inside <answer></answer> as exactly yes or no. Rules:\n"
        f"- Start from the strongest findings; a strong finding outweighs several weak ones.\n"
        f"- If findings conflict, say which side dominates and why.\n"
        f"- Refer only to the findings and references above; do not introduce new numbers "
        f"or facts.\n"
        f"- Weak findings such as low activity can reflect lifestyle rather than disease; "
        f"hedge accordingly.\n"
        f"Output nothing outside the two tags."
    )


_TAG_ANSWER_RE = re.compile(r"<answer>\s*(yes|no)\s*</answer>", re.IGNORECASE)
_ANSWER_RE = re.compile(r"answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_verdict(text: str, label: str) -> str | None:
    """Return 'positive' / 'negative' from <answer> or the final Answer line, or None."""
    tagged = _TAG_ANSWER_RE.findall(text)
    if tagged:
        return "positive" if tagged[-1].lower() == "yes" else "negative"
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    verdict = matches[-1].strip().lower()
    return "negative" if verdict.startswith("no ") or verdict == "no" else "positive"


# ── Ground-truth labels ──────────────────────────────────────────────────────

def _load_label_map(labels_file: str, label: str) -> dict[str, str]:
    """user_id -> 'positive' / 'negative' / 'unknown' for the target label.

    MHC encodes 1=positive, 0=negative, -1=unknown. Trait labels are constant per
    user, so we de-duplicate on user_id.
    """
    import pandas as pd

    df = pd.read_parquet(labels_file, columns=["user_id", label])
    df = df.dropna(subset=[label]).drop_duplicates("user_id")
    mapping = {"1": "positive", "1.0": "positive", "0": "negative", "0.0": "negative"}
    out: dict[str, str] = {}
    for uid, val in zip(df["user_id"], df[label]):
        out[uid] = mapping.get(str(val).strip(), "unknown")
    return out


# ── Record / signal-result iteration ─────────────────────────────────────────

@dataclass
class _RecordMeta:
    """Just the fields the rationale loop needs from a record."""
    row_id: str
    user_id: str
    date: str
    wear_pct: float | None = None


def _iter_recordings(args):
    if args.variant == "daily":
        from clinical.data import iter_daily_recordings
        return iter_daily_recordings(
            min_wear_pct=args.min_wear_pct,
            split_users=args.split_users,
            max_rows=args.max_rows,
            max_users=args.max_users,
        )
    from clinical.data import iter_weekly_recordings
    return iter_weekly_recordings(
        min_valid_hours=args.min_valid_hours,
        split_users=args.split_users,
        max_rows=args.max_rows,
        max_users=args.max_users,
    )


def _signal_results_from_annotations(anns_raw, relevant_signals: list[str]) -> dict:
    """Bucket pre-generated annotations into a {signal_name: [Annotation]} dict.

    Initialises every relevant signal to an empty list first, so absence signals
    (e.g. signal_17) still fire via the grouper when no annotation is present —
    mirroring the live path, which always runs the synthesizer.
    """
    from timef.schema import Annotation

    results: dict[str, list] = {name: [] for name in relevant_signals}
    for a in anns_raw:
        d = a if isinstance(a, dict) else json.loads(a)
        base = str(d.get("label") or "").split(":")[0]
        signal = BASE_LABEL_TO_SIGNAL.get(base)
        if signal not in results:  # not relevant to this label (or unknown)
            continue
        results[signal].append(Annotation(
            caption_type=d.get("caption_type", "cross_channel"),
            text=d.get("text", ""),
            label=d.get("label"),
        ))
    return results


def _iter_arrow(args, relevant_signals: list[str]):
    """Yield (_RecordMeta, signal_results) from pre-generated caption .arrow shards."""
    import glob
    import os

    import pyarrow as pa

    files = sorted(glob.glob(os.path.join(args.captions_dir, "*.arrow")))
    if not files:
        sys.exit(f"No .arrow files found in {args.captions_dir!r}.")

    seen = 0
    seen_users: set[str] = set()
    for path in files:
        with pa.memory_map(path, "r") as src:
            table = pa.ipc.open_file(src).read_all().select(
                ["row_id", "user_id", "date", "wear_pct", "annotations"]
            )
        for row in table.to_pylist():
            uid = row["user_id"]
            if args.split_users is not None and uid not in args.split_users:
                continue
            if args.max_users is not None:
                seen_users.add(uid)
                if len(seen_users) > args.max_users:
                    return
            if args.max_rows is not None and seen >= args.max_rows:
                return
            meta = _RecordMeta(row_id=row["row_id"], user_id=uid, date=str(row["date"]),
                               wear_pct=row.get("wear_pct"))
            signal_results = _signal_results_from_annotations(
                row["annotations"], relevant_signals
            )
            yield meta, signal_results
            seen += 1


def _iter_records_and_signals(args, relevant_signals: list[str], config):
    """Unified source: yields (meta, signal_results) from arrow files or live synthesizers."""
    if args.captions_dir:
        yield from _iter_arrow(args, relevant_signals)
        return
    synthesizers = {
        name: synth for name, synth in
        (_daily_synthesizers() if args.variant == "daily" else _weekly_synthesizers()).items()
        if name in relevant_signals
    }
    for rec in _iter_recordings(args):
        signal_results = {name: synth.synthesize(rec, config)
                          for name, synth in synthesizers.items()}
        yield rec, signal_results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate clinical rationales from the caption pipeline via an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant", choices=["daily", "weekly"], default="daily")
    p.add_argument("--captions_dir", default=None,
                   help="Load pre-generated captions from a dir of .arrow shards "
                        "(e.g. clinical_captions_20users) instead of re-running the "
                        "synthesizers live. Note: arrow captions are a frozen snapshot — "
                        "they reflect the synthesizer code at export time, not current edits.")
    p.add_argument("--label", required=True,
                   help="Target label column (e.g. cardiovascular_disease, 'Heart Failure or CHF').")
    p.add_argument("--labels_file", required=True,
                   help="Parquet with user_id + label columns for ground truth (e.g. weekly_labels.parquet).")
    p.add_argument("--out", required=True, help="Output JSONL path.")
    p.add_argument("--model", default="gpt-4o", help="Model id for the OpenAI-compatible endpoint.")
    p.add_argument("--base_url", default=None, help="Override endpoint (else OPENAI_BASE_URL env / OpenAI).")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--style", choices=sorted(STYLE_INSTRUCTIONS) + ["structured"], default="reconsider",
                   help="'structured': pipeline-built <findings> (direction + strength) and an "
                        "LLM-written <reasoning> + <answer>.")
    p.add_argument("--keep_correct", action="store_true",
                   help="Write only rationales whose verdict matches the ground truth (the "
                        "label is never shown to the model).")
    p.add_argument("--max_rows", type=int, default=None, help="Cap recordings processed.")
    p.add_argument("--max_users", type=int, default=None, help="Cap unique participants.")
    p.add_argument("--min_wear_pct", type=float, default=50.0, help="Daily wear filter.")
    p.add_argument("--min_valid_hours", type=int, default=0, help="Weekly coverage filter.")
    p.add_argument("--split_file", default=None,
                   help="Restrict to sharable users (JSON train/val/test or parquet with user_id).")
    p.add_argument("--skip_unknown_gt", action="store_true",
                   help="Skip recordings whose ground-truth label is unknown (-1).")
    p.add_argument("--top_quality", type=int, default=None,
                   help="Select only the N highest data-quality recordings: a first pass "
                        "scores every candidate (no API calls), then the LLM runs on the "
                        "top N. Ranked by wear_pct, gated by --min_fired and known ground "
                        "truth.")
    p.add_argument("--min_fired", type=int, default=4,
                   help="In --top_quality mode, require at least this many distinct signals "
                        "to have fired (enough evidence for a meaningful rationale).")
    p.add_argument("--per_user", type=int, default=1,
                   help="In --top_quality mode, recordings kept per participant.")
    p.add_argument("--balance", action="store_true",
                   help="In --top_quality mode, take half of N from each class.")
    return p.parse_args()


def _load_split_users(split_file: str) -> set[str]:
    path = Path(split_file)
    if path.suffix == ".parquet":
        import pandas as pd
        return set(pd.read_parquet(path)["user_id"].dropna().unique().tolist())
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return set(data)
    return set(data.get("train", [])) | set(data.get("validation", [])) | set(data.get("test", []))


def main() -> None:
    args = _parse_args()
    args.split_users = _load_split_users(args.split_file) if args.split_file else None

    if args.label not in {lbl for lbls in SIGNAL_LABEL_RELEVANCE.values() for lbl in lbls}:
        sys.exit(f"No signals are mapped to label {args.label!r}. "
                 f"Check clinical.caption_grouper.SIGNAL_LABEL_RELEVANCE.")

    variant_signals = DAILY_SIGNAL_NAMES if args.variant == "daily" else WEEKLY_SIGNAL_NAMES
    relevant_signals = [
        name for name in variant_signals
        if args.label in SIGNAL_LABEL_RELEVANCE.get(name, [])
    ]
    if not relevant_signals:
        sys.exit(f"No {args.variant} signals are relevant to {args.label!r}.")

    from clinical.data import DAILY_CLINICAL_CONFIG, WEEKLY_CLINICAL_CONFIG
    config = DAILY_CLINICAL_CONFIG if args.variant == "daily" else WEEKLY_CLINICAL_CONFIG
    window_desc = ("the last 24 hours at minute resolution" if args.variant == "daily"
                   else "the last 7 days at hourly resolution")

    label_map = _load_label_map(args.labels_file, args.label)

    model = ClientModel(
        config=ClientConfig(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_requests=10**9,  # effectively uncapped; we run a full sweep
        ),
        channel_config=config,  # unused for text-only calls, but required by the base
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source = f"arrow ({args.captions_dir})" if args.captions_dir else "live synthesizers"
    print(f"Label    : {args.label} ({_display_label(args.label)})")
    print(f"Variant  : {args.variant}  |  source: {source}")
    print(f"Signals  : {', '.join(sorted(relevant_signals))}")
    print(f"Model    : {args.model}  |  style: {args.style}")
    print(f"Output   : {out_path}\n")

    n_skipped = 0

    def _candidates():
        """Yield (rec, caps, fired, gt, signal_results) for every record with non-empty captions."""
        nonlocal n_skipped
        for rec, signal_results in _iter_records_and_signals(args, relevant_signals, config):
            grouped = group_captions(signal_results, labels=[args.label])
            caps = grouped[args.label]
            if caps.is_empty():
                n_skipped += 1
                continue
            gt = label_map.get(rec.user_id, "unknown")
            fired = _fired_signals(signal_results, args.label)
            yield rec, caps, fired, gt, signal_results

    # In --top_quality mode, score every candidate first (no API calls), keep the best
    # --per_user recordings per user, then take the best N (half per class with --balance).
    # Otherwise stream all candidates as they come.
    if args.top_quality:
        per_user: dict[str, list[tuple]] = {}
        for cand in _candidates():
            rec, _caps, fired, gt, _sr = cand
            if gt == "unknown" or len(fired) < args.min_fired:
                n_skipped += 1
                continue
            # Rank by sensor completeness, then evidence richness.
            wear = rec.wear_pct if rec.wear_pct is not None else 0.0
            per_user.setdefault(rec.user_id, []).append(((wear, len(fired)), cand))
        kept = [entry for entries in per_user.values()
                for entry in sorted(entries, key=lambda x: x[0], reverse=True)[:args.per_user]]
        scored = sorted(kept, key=lambda x: x[0], reverse=True)
        if args.balance:
            half = args.top_quality // 2
            pos = [e for e in scored if e[1][3] == "positive"][:half]
            neg = [e for e in scored if e[1][3] == "negative"][:args.top_quality - len(pos)]
            selected = pos + neg
        else:
            selected = scored[:args.top_quality]
        worklist = [cand for _, cand in selected]
        n_pos = sum(1 for _, cand in selected if cand[3] == "positive")
        print(f"Selected {len(worklist)} of {len(scored)} eligible recordings "
              f"(<= {args.per_user} per participant; >= {args.min_fired} signals fired, known GT), "
              f"ranked by wear_pct. Class balance: {n_pos} positive / "
              f"{len(worklist) - n_pos} negative.")
        for (wear, nf), cand in selected:
            print(f"  {cand[0].row_id:<40} wear={wear:5.1f}%  signals={nf}  gt={cand[3]}")
        print()
    else:
        worklist = _candidates()

    n_written = n_correct = n_scored = n_rejected = 0
    tok_in = tok_out = 0

    with out_path.open("w") as fh:
        for rec, caps, fired, gt, signal_results in worklist:
            if not args.top_quality and args.skip_unknown_gt and gt == "unknown":
                n_skipped += 1
                continue

            findings = None
            if args.style == "structured":
                findings = findings_for(signal_results, args.label)
                prompt = build_structured_prompt(args.label, findings, window_desc)
            else:
                prompt = build_prompt(args.label, caps, fired, window_desc, args.style)
            resp = model.process(prompt, signals=[])  # text-only: no plots
            rationale = (f"{render_findings(findings, args.label)}\n{resp.text.strip()}"
                         if findings is not None else resp.text)

            predicted = parse_verdict(resp.text, args.label)
            tok_in += resp.input_tokens
            tok_out += resp.output_tokens
            if gt in ("positive", "negative") and predicted is not None:
                n_scored += 1
                n_correct += int(predicted == gt)
            if args.keep_correct and predicted != gt:
                n_rejected += 1
                continue

            fh.write(json.dumps({
                "row_id": rec.row_id,
                "user_id": rec.user_id,
                "date": rec.date,
                "wear_pct": rec.wear_pct,
                "label": args.label,
                "ground_truth": gt,
                "predicted": predicted,
                "fired_signals": fired,
                "indicating": caps.indicating,
                "not_indicating": caps.not_indicating,
                "findings": ([{"signal": f.signal, "text": f.text, "supports": f.supports,
                               "strength": f.strength} for f in findings]
                             if findings is not None else None),
                "rationale": rationale,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
            }) + "\n")
            fh.flush()
            n_written += 1
            if n_written % 25 == 0:
                acc = f"{n_correct / n_scored:.1%}" if n_scored else "n/a"
                print(f"  {n_written} written | running acc {acc} | "
                      f"tok {tok_in}+{tok_out}")

    acc = f"{n_correct / n_scored:.1%} ({n_correct}/{n_scored})" if n_scored else "n/a"
    print(f"\nDone. Wrote {n_written} rationales, skipped {n_skipped} "
          f"(no relevant captions / unknown GT), rejected {n_rejected} with a wrong verdict.")
    print(f"Verdict accuracy vs. ground truth: {acc}")
    print(f"Tokens: {tok_in} in, {tok_out} out.")


if __name__ == "__main__":
    main()
