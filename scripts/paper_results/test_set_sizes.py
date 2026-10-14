#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Per-task test-set composition (n_users, n_positive, level distributions).

Reads ground-truth columns from the longest available prediction CSV for each
task (any model run shares the test split, so the longest CSV reflects the
full evaluable population) and emits:

* For binary tasks: n_users, n_positive (minority-class count, matching the
  AUPRC reporting convention), and minority prevalence.
* For ordinal tasks: n_users plus the per-level user count and percentage.

Outputs are written next to the script as ``test_set_sizes.{json,md}`` so the
numbers stay version-controlled alongside the code that produced them.

Run::

    python scripts/paper_results/test_set_sizes.py \\
        --predictions_dir ~/Downloads/results \\
        --secondary_dir "~/Downloads/results 2"
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# Task configuration


@dataclass(frozen=True)
class BinaryTaskConfig:
    display: str
    eval_positive: str  # the codebase's positive_label (used for F1/accuracy)


@dataclass(frozen=True)
class OrdinalTaskConfig:
    display: str
    levels: tuple[str, ...]


# Held-out clinical conditions (zero-shot evaluation). Used to annotate the
# Markdown output with daggers.
HELD_OUT = {"Diabetes", "cardiovascular_disease", "CAD", "Atrial_fibrillation__Afib_"}


BINARY_TASKS: dict[str, BinaryTaskConfig] = {
    "BiologicalSex":              BinaryTaskConfig("Biological Sex",          "Male"),
    "Hypertension":               BinaryTaskConfig("Hypertension",            "True"),
    "work":                       BinaryTaskConfig("Employment",              "True"),
    "sleep_diagnosis1":           BinaryTaskConfig("Sleep Diagnosis",         "True"),
    "Diabetes":                   BinaryTaskConfig("Diabetes",                "True"),
    "cardiovascular_disease":     BinaryTaskConfig("Cardiovascular Disease",  "True"),
    "CAD":                        BinaryTaskConfig("Coronary Artery Disease", "True"),
    "Atrial_fibrillation__Afib_": BinaryTaskConfig("Atrial Fibrillation",     "True"),
}

ORDINAL_TASKS: dict[str, OrdinalTaskConfig] = {
    "BMI_categories": OrdinalTaskConfig(
        "BMI (5 levels)",
        ("Underweight", "Normal weight", "Overweight", "Obesity", "Morbid Obesity"),
    ),
    "happiness_categories": OrdinalTaskConfig(
        "Happiness (4 levels)",
        ("Low", "Medium", "High", "Very High"),
    ),
    "blood_pressure_categories": OrdinalTaskConfig(
        "Blood Pressure (5 levels)",
        ("Normal", "Elevated", "Hypertension_Stage_1", "Hypertension_Stage_2", "Hypertensive_Crisis"),
    ),
    "sleep_time_categories": OrdinalTaskConfig(
        "Sleep Time (4 levels)",
        ("Insufficient", "Short", "Normal", "Too Long"),
    ),
}


# Helpers


def _find_longest_predictions_csv(task_dir: Path) -> Path | None:
    """Return the predictions.csv with the most rows under task_dir."""
    matches = list(task_dir.rglob("predictions.csv"))
    if not matches:
        return None
    return max(matches, key=lambda p: len(pd.read_csv(p)))


def _resolve_csv(task_name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        csv = _find_longest_predictions_csv(root / task_name)
        if csv is not None:
            return csv
    return None


# Computation


def compute_binary(csv: Path, cfg: BinaryTaskConfig) -> dict:
    """User-level minority-class size and prevalence."""
    df = pd.read_csv(csv)
    per_user = df.groupby("user_id")["ground_truth"].first().astype(str)
    n_users = int(per_user.size)

    counts = per_user.value_counts().to_dict()
    minority_label = min(counts, key=counts.get)
    n_minority = int(counts[minority_label])
    minority_prev = n_minority / n_users if n_users else 0.0

    return {
        "task": cfg.display,
        "type": "binary",
        "eval_positive_label": cfg.eval_positive,
        "minority_label": minority_label,
        "minority_matches_eval_positive": minority_label == cfg.eval_positive,
        "n_users": n_users,
        "n_positive_minority": n_minority,
        "minority_prevalence": minority_prev,
    }


def compute_ordinal(csv: Path, cfg: OrdinalTaskConfig) -> dict:
    """User-level per-ordinal-level distribution."""
    df = pd.read_csv(csv)
    per_user = df.groupby("user_id")["ground_truth"].first().astype(str)
    valid = per_user.isin(cfg.levels)
    per_user = per_user[valid]
    n_users = int(per_user.size)

    distribution = []
    for level in cfg.levels:
        n = int((per_user == level).sum())
        pct = n / n_users if n_users else 0.0
        distribution.append({"level": level, "n": n, "pct": pct})

    return {
        "task": cfg.display,
        "type": "ordinal",
        "levels": list(cfg.levels),
        "n_users": n_users,
        "distribution": distribution,
    }


# Output formatting


def format_markdown(records: list[dict]) -> str:
    binary = [r for r in records if r["type"] == "binary"]
    ordinal = [r for r in records if r["type"] == "ordinal"]

    out: list[str] = ["# Test-set composition\n"]
    out.append(
        "Per-task user counts on the captioned test partition. Binary tasks\n"
        "report the minority-class user count (matching the AUPRC reporting\n"
        "convention used in the paper). Ordinal tasks report the per-level\n"
        "distribution; uneven distributions reduce the effective signal in\n"
        "Spearman~ρ and QWK.\n"
    )

    out.append("\n## Binary tasks\n")
    out.append(
        "| Task | minority label | n\\_users | n\\_pos | prev (%) |"
    )
    out.append("|---|---|---:|---:|---:|")
    for r in binary:
        held_out_marker = "$^{\\dagger}$" if r["task"] in {
            "Diabetes", "Cardiovascular Disease",
            "Coronary Artery Disease", "Atrial Fibrillation",
        } else ""
        out.append(
            f"| {r['task']}{held_out_marker} | {r['minority_label']} | "
            f"{r['n_users']:,} | {r['n_positive_minority']:,} | "
            f"{r['minority_prevalence']*100:.1f} |"
        )

    out.append("\n## Ordinal tasks\n")
    out.append("| Task | n\\_users | Level distribution (n, %) |")
    out.append("|---|---:|---|")
    for r in ordinal:
        dist_str = "; ".join(
            f"{d['level']} {d['n']} ({d['pct']*100:.1f}%)"
            for d in r["distribution"]
        )
        out.append(f"| {r['task']} | {r['n_users']:,} | {dist_str} |")

    return "\n".join(out) + "\n"


# CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=Path.home() / "Downloads" / "results",
        help="Primary directory of prediction CSVs (one subdir per task).",
    )
    parser.add_argument(
        "--secondary_dir",
        type=Path,
        default=Path.home() / "Downloads" / "results 2",
        help="Optional secondary directory (held-out clinical / ordinal tasks).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Directory to write test_set_sizes.{json,md} into. Defaults to the "
            "directory of this script so the snapshot stays version-controlled "
            "alongside it (results/ is gitignored)."
        ),
    )
    args = parser.parse_args()

    roots = [args.predictions_dir, args.secondary_dir]
    records: list[dict] = []

    for task_name, cfg in BINARY_TASKS.items():
        csv = _resolve_csv(task_name, roots)
        if csv is None:
            print(f"[skip] {cfg.display}: no predictions.csv found")
            continue
        records.append(compute_binary(csv, cfg))

    for task_name, cfg in ORDINAL_TASKS.items():
        csv = _resolve_csv(task_name, roots)
        if csv is None:
            print(f"[skip] {cfg.display}: no predictions.csv found")
            continue
        records.append(compute_ordinal(csv, cfg))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "test_set_sizes.json"
    md_path = args.output_dir / "test_set_sizes.md"

    json_path.write_text(json.dumps(records, indent=2) + "\n")
    md_path.write_text(format_markdown(records))

    print(format_markdown(records))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
