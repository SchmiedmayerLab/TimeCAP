#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Collect the caption-family ablation grid into a comparison table.

Reads the same results tree ``gpu_sweep.py`` writes and reuses its
protocol-aware ``find_completed_run``, so a cell is populated only by a run
this sweep would itself consider complete — full test set, matching
``max_new_tokens``, matching split/labels/export paths.

    python scripts/collect_ablation_results.py --output ABLATION_RESULTS.md

The headline metric is chosen from the keys present in metrics.json rather
than by re-deriving each task's label type: binary tasks carry ``f1``,
ordinal ``qwk``, multiclass ``macro_f1``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.gpu_sweep import DEFAULT_CHECKPOINTS, Job, find_completed_run  # noqa: E402

TASKS = [
    "afib", "bmi", "biological_sex", "cad", "cardiovascular",
    "diabetes", "happiness", "hypertension", "sleep_diagnosis", "work",
]

# Order matters: the first key present wins, and binary tasks also carry
# ``accuracy``, so ``f1`` has to be checked before it.
HEADLINE_KEYS = ["f1", "qwk", "macro_f1"]


def headline(metrics: dict) -> tuple[str, float, float] | None:
    for key in HEADLINE_KEYS:
        if key in metrics:
            return key, float(metrics[key]), float(metrics.get(f"{key}_se", 0.0))
    return None


def trivial_headline(metrics: dict, key: str) -> float | None:
    """What the *headline* metric alone gives a constant answer.

    Useful context, but not a skill test: on a task whose positive class is the
    majority, always-positive scores a high F1, and a model with real skill can
    land on the same F1 from a different precision/recall trade. Read this
    column next to ``skill_check``, never instead of it.

    * binary: always-positive has precision = prevalence, recall = 1, so
      F1 = 2p / (1 + p).
    * ordinal: a constant prediction has zero variance, so QWK = 0.
    * multiclass: needs the per-class distribution, which metrics.json does not
      carry, so report nothing rather than guess.
    """
    if key == "f1":
        n = metrics.get("n_samples", 0)
        positives = metrics.get("n_positive_gt", 0)
        if n:
            p = positives / n
            return 2 * p / (1 + p) if p else 0.0
        return None
    if key == "qwk":
        return 0.0
    return None


def skill_check(metrics: dict, key: str) -> tuple[float, float, float] | None:
    """(model, trivial, se) on a metric a constant answer cannot game.

    F1 is the headline here because the task defines it that way, but it is the
    wrong instrument for "did this learn anything": it ignores true negatives,
    so a model can beat always-positive F1 while being far worse than always-
    negative overall. Accuracy against the majority-class rate is the honest
    test, and it is what decides the skill verdict.

    Returns None for multiclass, where metrics.json lacks the class counts
    needed to state a majority rate.
    """
    if key == "f1":
        n = metrics.get("n_samples", 0)
        if not n:
            return None
        p = metrics.get("n_positive_gt", 0) / n
        return (
            float(metrics.get("accuracy", 0.0)),
            max(p, 1 - p),
            float(metrics.get("accuracy_se", 0.0)),
        )
    if key == "qwk":
        return float(metrics["qwk"]), 0.0, float(metrics.get("qwk_se", 0.0))
    return None


def bundle_artifacts(run_dirs: dict[tuple[str, str], Path], dest: Path) -> None:
    """Copy each cell's artifacts into one flat, self-contained directory.

    The results tree is already on the shared mount and survives the node, but
    it is keyed by display label and timestamp, so finding the run that backs a
    given (checkpoint, task) means re-deriving the same lookup this script
    does. The bundle is that lookup, materialised once.
    """
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for (stem, task), run_dir in sorted(run_dirs.items()):
        cell_dir = dest / f"{stem}__{task}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.json", "run_config.json", "predictions.csv"):
            src = run_dir / name
            if src.exists():
                shutil.copy2(src, cell_dir / name)
                copied += 1
        # Where the artifacts came from, so a bundle is traceable back to the
        # run that produced it without guessing at timestamps.
        (cell_dir / "SOURCE").write_text(f"{run_dir}\n")

    sweep_logs = REPO_ROOT / "sweep_logs"
    if sweep_logs.is_dir():
        shutil.copytree(sweep_logs, dest / "sweep_logs", dirs_exist_ok=True)

    print(f"Bundled {copied} files from {len(run_dirs)} runs into {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--bundle-dir", type=Path, default=None,
        help="Also copy every cell's metrics.json, run_config.json and "
             "predictions.csv here, plus the sweep logs. ~180 MB for the full "
             "40-cell grid.",
    )
    args = parser.parse_args()

    ckpt_dir = (
        REPO_ROOT / "results/meta-llama_Llama-3.2-1B/OpenTSLMSP"
        / "stage_joint_caption_metabolic/checkpoints"
    )

    cells: dict[tuple[str, str], dict] = {}
    run_dirs: dict[tuple[str, str], Path] = {}
    for stem in args.checkpoints:
        for task in args.tasks:
            job = Job(ckpt_dir / f"{stem}.pt", task, "meta-llama/Llama-3.2-1B")
            run_dir = find_completed_run(
                args.results_dir, task, job.model_dir_name, args
            )
            if run_dir is None:
                continue
            metrics = json.loads((run_dir / "metrics.json").read_text())
            cells[(stem, task)] = metrics
            run_dirs[(stem, task)] = run_dir

    if args.bundle_dir:
        bundle_artifacts(run_dirs, args.bundle_dir)

    lines: list[str] = []
    add = lines.append

    n_expected = len(args.checkpoints) * len(args.tasks)
    add(f"# Caption-family ablation — {len(cells)}/{n_expected} cells complete\n")

    parse_rates = [m.get("parse_rate", float("nan")) for m in cells.values()]
    if parse_rates:
        add(f"Parse rate across all cells: "
            f"**{min(parse_rates):.4f} – {max(parse_rates):.4f}**\n")

    short = {
        c: c.replace("joint_", "").replace("_1b_final", "").replace("_final_1b", "")
        for c in args.checkpoints
    }

    add("| task | metric | " + " | ".join(short[c] for c in args.checkpoints)
        + " | trivial | spread | verdict |")
    add("|---|---|" + "---|" * (len(args.checkpoints) + 3))

    flagged: list[str] = []
    no_skill: list[str] = []
    skill_rows: list[str] = []
    for task in args.tasks:
        row_metrics = [cells.get((c, task)) for c in args.checkpoints]
        present = [m for m in row_metrics if m]
        if not present:
            add(f"| {task} | — | " + " | ".join("—" for _ in args.checkpoints)
                + " | — | — | — |")
            continue

        key = headline(present[0])[0]
        vals, ses, texts = [], [], []
        for m in row_metrics:
            if not m:
                texts.append("—")
                continue
            _, v, se = headline(m)
            vals.append(v)
            ses.append(se)
            texts.append(f"{v:.4f} ± {se:.4f}")

        spread = max(vals) - min(vals)
        # Difference of two independent estimates: se_diff = sqrt(se_a^2+se_b^2).
        # Compare the full spread against the widest such interval in the row,
        # which is the relevant question for "are these four distinguishable".
        se_diff = (max(ses) ** 2 + sorted(ses)[-2] ** 2) ** 0.5 if len(ses) > 1 else 0.0
        noise = spread < 2 * se_diff

        base = trivial_headline(present[0], key)
        if noise:
            flagged.append(task)
        add(f"| {task} | {key} | " + " | ".join(texts)
            + f" | {'—' if base is None else f'{base:.4f}'}"
            + f" | {spread:.4f} | {'**within noise**' if noise else 'separated'} |")

        # Skill is a separate question from separation, and answering them in
        # one column is what made an earlier version call biological_sex
        # degenerate on an F1 tie while its accuracy beat the majority rate.
        checks = [(c, skill_check(m, key)) for c, m in
                  zip(args.checkpoints, row_metrics) if m]
        checks = [(c, s) for c, s in checks if s]
        if checks:
            best_c, (best_v, triv, best_se) = max(checks, key=lambda cs: cs[1][0])
            delta = best_v - triv
            sigmas = delta / best_se if best_se else 0.0
            label = "accuracy" if key == "f1" else key
            skill_rows.append(
                f"| {task} | {label} | {best_v:.4f} ({short[best_c]}) | "
                f"{triv:.4f} | {delta:+.4f} | {sigmas:+.1f}σ | "
                f"{'yes' if delta > 0 and sigmas > 2 else '**no**'} |"
            )
            if not (delta > 0 and sigmas > 2):
                no_skill.append(task)

    add("")
    add("The `trivial` column is the headline metric's own constant-answer "
        "score. It is context, not a skill test — F1 ignores true negatives, so "
        "clearing it does not mean a model learned anything. The next table is "
        "the actual test.\n")

    if skill_rows:
        add("## Does anything beat a constant answer?\n")
        add("Best checkpoint per task on a metric a constant answer cannot "
            "game: accuracy vs. the majority-class rate (binary), QWK vs. 0 "
            "(ordinal). Multiclass tasks are omitted — metrics.json lacks the "
            "class counts needed to state a majority rate.\n")
        add("| task | metric | best | trivial | delta | significance | beats trivial? |")
        add("|---|---|---|---|---|---|---|")
        lines.extend(skill_rows)
        add("")
        if no_skill:
            add(f"**No skill over trivial:** {', '.join(f'`{t}`' for t in no_skill)}"
                " — caption-family differences on these tasks are differences "
                "between models that do not beat a constant answer.\n")

    if flagged:
        add("## Tasks where the four checkpoints are within noise\n")
        add(", ".join(f"`{t}`" for t in flagged))
        add("\nSpread is smaller than 2x the standard error of a pairwise "
            "difference, so the caption families are not distinguishable here.\n")

    add("## Sample counts\n")
    add("| task | n_evaluated | parse_rate range |")
    add("|---|---|---|")
    for task in args.tasks:
        ms = [cells[(c, task)] for c in args.checkpoints if (c, task) in cells]
        if not ms:
            continue
        ns = {m.get("n_evaluated") for m in ms}
        prs = [m.get("parse_rate", 0.0) for m in ms]
        add(f"| {task} | {'/'.join(str(n) for n in sorted(ns, key=str))} | "
            f"{min(prs):.4f} – {max(prs):.4f} |")

    text = "\n".join(lines)
    if args.output:
        args.output.write_text(text)
        print(f"Wrote {args.output}")
    if args.bundle_dir:
        (args.bundle_dir / "ABLATION_RESULTS.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
