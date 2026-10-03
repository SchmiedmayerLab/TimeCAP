#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Build a LaTeX report pairing generated rationales with their signal plots.

Reads the JSONL produced by ``scripts/generate_rationales.py`` (each line carries
row_id, the indicating / not-indicating captions, the LLM rationale, ground truth and
prediction), re-reads the raw values for those exact recordings from the caption
``.arrow`` shards, renders the 3-panel plot for each (reusing ``generate_cvd_report.plot_row``),
and emits ``<out_dir>/report.tex`` plus ``<out_dir>/plots/*.pdf``.

Example::

    python scripts/build_rationale_report.py \\
        --jsonl rationales/cvd_top10.jsonl \\
        --captions_dir clinical_captions_20users \\
        --out_dir reports/cvd_top10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa

from generate_cvd_report import plot_row  # reuse the demo's 3-panel renderer


# ── LaTeX escaping ────────────────────────────────────────────────────────────

_ESCAPES = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
    ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
    ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
]
# Applied after escaping, so the LaTeX they introduce (backslashes, $) is intentional.
_UNICODE = [
    ("≥", r"$\ge$"), ("≤", r"$\le$"), ("≈", r"$\approx$"), ("∼", r"$\sim$"),
    ("–", "--"), ("—", "---"), ("×", r"$\times$"), ("→", r"$\rightarrow$"),
    ("°", r"$^\circ$"), ("µ", r"$\mu$"),
]


def tex(s: str) -> str:
    for a, b in _ESCAPES:
        s = s.replace(a, b)
    for a, b in _UNICODE:
        s = s.replace(a, b)
    return s


# ── Arrow lookup ──────────────────────────────────────────────────────────────

def _load_records(captions_dir: str, row_ids: set[str]) -> dict[str, dict]:
    """row_id -> {user_id, date, channel_names, values} for the requested ids only."""
    out: dict[str, dict] = {}
    for shard in sorted(glob.glob(os.path.join(captions_dir, "*.arrow"))):
        with pa.memory_map(shard, "r") as src:
            table = pa.ipc.open_file(src).read_all()
        ids = table.column("row_id").to_pylist()
        idxs = [i for i, rid in enumerate(ids) if rid in row_ids and rid not in out]
        if not idxs:
            continue
        sub = table.take(idxs).select(
            ["row_id", "user_id", "date", "channel_names", "values"]
        )
        for r in sub.to_pylist():
            out[r["row_id"]] = r
    return out


# ── LaTeX assembly ────────────────────────────────────────────────────────────

def _items(captions: list[str]) -> str:
    if not captions:
        return r"  \item \textit{(none)}"
    return "\n".join(f"  \\item {tex(c)}" for c in captions)


def _split_answer(rationale: str) -> tuple[str, str | None]:
    """Separate the trailing 'Answer: ...' line from the rationale body."""
    m = list(re.finditer(r"Answer\s*:\s*(.+?)\s*$", rationale, re.IGNORECASE | re.MULTILINE))
    if not m:
        return rationale.strip(), None
    last = m[-1]
    body = rationale[:last.start()].strip()
    return body, last.group(1).strip()


def _section(idx: int, rec: dict, entry: dict, disp: str, fig_rel: str) -> str:
    gt = entry.get("ground_truth", "unknown")
    pred = entry.get("predicted")
    correct = (pred == gt) if gt in ("positive", "negative") and pred else None
    verdict_note = (
        "correct" if correct else "incorrect" if correct is False else "n/a"
    )
    wear = entry.get("wear_pct")
    wear_str = f"{wear:.0f}\\%" if isinstance(wear, (int, float)) else "n/a"

    body, answer = _split_answer(entry.get("rationale", ""))
    body_latex = tex(body).replace("\n\n", "\n\n\\noindent ")
    answer_line = (
        f"\\textbf{{Answer: {tex(answer)}}}" if answer else ""
    )

    return textwrap.dedent(f"""\
        \\subsection*{{Case {idx}: {tex(disp)}}}
        \\texttt{{{tex(rec['user_id'][:14])}...}}\\quad Date: {tex(str(rec['date']))}\\quad
        Wear: {wear_str}\\quad
        Ground truth: \\textbf{{{tex(gt)}}}\\quad
        Predicted: \\textbf{{{tex(str(pred))}}} ({verdict_note})

        \\begin{{figure}}[h]
          \\centering
          \\includegraphics[width=\\textwidth]{{{fig_rel}}}
        \\end{{figure}}

        \\textbf{{Indicating {tex(disp)}:}}
        \\begin{{itemize}}
        {_items(entry.get('indicating', []))}
        \\end{{itemize}}

        \\textbf{{Not indicating {tex(disp)}:}}
        \\begin{{itemize}}
        {_items(entry.get('not_indicating', []))}
        \\end{{itemize}}

        \\textbf{{Rationale (model-generated):}}\\par
        {body_latex}

        \\par\\medskip {answer_line}

        \\vspace{{1em}}\\hrule\\vspace{{1em}}
        """)


def _document(title: str, intro: str, sections: list[str]) -> str:
    body = "\n".join(sections)
    return textwrap.dedent(f"""\
        \\documentclass[11pt,a4paper]{{article}}
        \\usepackage[margin=2.5cm]{{geometry}}
        \\usepackage[utf8]{{inputenc}}
        \\usepackage{{amssymb}}
        \\usepackage{{graphicx}}
        \\usepackage{{parskip}}
        \\usepackage{{microtype}}
        \\title{{{title}}}
        \\author{{SensorTSLM Clinical Pipeline}}
        \\date{{}}
        \\begin{{document}}
        \\maketitle

        \\noindent {intro}

        \\bigskip\\hrule\\bigskip

        {body}
        \\end{{document}}
        """)


# ── Main ─────────────────────────────────────────────────────────────────────

DISPLAY_LABELS = {
    "cardiovascular_disease": "cardiovascular disease",
    "CAD": "coronary artery disease (CAD)",
    "Heart Failure or CHF": "heart failure (CHF)",
    "Hypertension": "hypertension",
    "Diabetes": "diabetes",
    "sleep_diagnosis1": "a sleep disorder",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Build a LaTeX rationale+plot report.")
    p.add_argument("--jsonl", required=True, help="Rationale JSONL from generate_rationales.py.")
    p.add_argument("--captions_dir", required=True, help="Dir of caption .arrow shards (raw values).")
    p.add_argument("--out_dir", required=True, help="Output dir for report.tex and plots/.")
    p.add_argument("--title", default=None, help="Report title (default derived from label).")
    args = p.parse_args()

    entries = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    if not entries:
        sys.exit(f"No rows in {args.jsonl}.")

    label = entries[0].get("label", "")
    disp = DISPLAY_LABELS.get(label, label.replace("_", " "))
    title = args.title or f"Generated Rationales: {disp.title()}"

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(args.captions_dir, {e["row_id"] for e in entries})

    sections: list[str] = []
    missing: list[str] = []
    for idx, entry in enumerate(entries, 1):
        rid = entry["row_id"]
        rec = records.get(rid)
        if rec is None:
            missing.append(rid)
            continue
        safe = re.sub(r"[^A-Za-z0-9]+", "_", rid)
        fig_path = plots_dir / f"{safe}.pdf"
        plot_row(rec, fig_path)
        sections.append(_section(idx, rec, entry, disp, f"plots/{fig_path.name}"))

    n_correct = sum(1 for e in entries
                    if e.get("predicted") == e.get("ground_truth")
                    and e.get("ground_truth") in ("positive", "negative"))
    n_scored = sum(1 for e in entries if e.get("ground_truth") in ("positive", "negative"))
    intro = (
        f"Each case shows one day of wearable sensor data for a single participant, "
        f"the clinical signal captions grouped into those \\textbf{{indicating}} and "
        f"\\textbf{{not indicating}} {tex(disp)}, and the model-generated rationale that "
        f"weighs them. Verdict accuracy on this set: {n_correct}/{n_scored}."
    )

    tex_path = out_dir / "report.tex"
    tex_path.write_text(_document(title, intro, sections))

    print(f"Wrote {tex_path} with {len(sections)} cases ({len(records)} plots).")
    if missing:
        print(f"WARNING: {len(missing)} row_ids not found in {args.captions_dir}: {missing}")
    print(f"Compile with:  cd {out_dir} && pdflatex report.tex")


if __name__ == "__main__":
    main()
