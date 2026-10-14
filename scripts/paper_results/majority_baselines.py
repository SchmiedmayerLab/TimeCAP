#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Majority-class baselines for the metabolic + held-out evaluation tasks.

Reads ground-truth columns from existing prediction CSVs (any model run, since
the test split is identical across models) and computes the metrics that an
always-predict-majority predictor would achieve on the same population.

Outputs one Markdown + one JSON file under ``results/paper/`` so the numbers
stay version-controlled alongside the code that produced them.

Run::

    python scripts/paper_results/majority_baselines.py \\
        --predictions_dir ~/Downloads/results \\
        --secondary_dir "~/Downloads/results 2" \\
        --output_dir results/paper

The script is purely additive (reads prediction CSVs, writes summary files);
nothing in the eval pipeline is modified.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


# Task configuration


@dataclass(frozen=True)
class BinaryTaskConfig:
    display: str
    positive: str  # canonical positive label string


@dataclass(frozen=True)
class OrdinalTaskConfig:
    display: str
    levels: tuple[str, ...]  # ordered low -> high


# Task directory name -> config. Each task is rendered as one row in the
# output tables. The "positive" / "levels" entries match
# ``time_series_datasets.mhc_label_lookup.METABOLIC_LABEL_CONFIG`` so the
# per-class scoring is identical to what the eval pipeline emits.
BINARY_TASKS: dict[str, BinaryTaskConfig] = {
    "BiologicalSex":           BinaryTaskConfig("Biological Sex",         "Male"),
    "Hypertension":            BinaryTaskConfig("Hypertension",           "True"),
    "work":                    BinaryTaskConfig("Employment",             "True"),
    "sleep_diagnosis1":        BinaryTaskConfig("Sleep Diagnosis",        "True"),
    "Diabetes":                BinaryTaskConfig("Diabetes",               "True"),
    "cardiovascular_disease":  BinaryTaskConfig("Cardiovascular Disease", "True"),
    "CAD":                     BinaryTaskConfig("Coronary Artery Disease", "True"),
    "Atrial_fibrillation__Afib_": BinaryTaskConfig("Atrial Fibrillation", "True"),
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


# Computation


def _find_predictions_csv(task_dir: Path) -> Path | None:
    """Return the predictions.csv with the most rows under task_dir."""
    matches = list(task_dir.rglob("predictions.csv"))
    if not matches:
        return None
    # Parse each candidate to count true rows (``raw_prediction`` can contain
    # embedded newlines, so wc-style line counting overestimates). The CSVs
    # are small enough that re-parsing the chosen one downstream is cheap.
    return max(matches, key=lambda p: len(pd.read_csv(p)))


def compute_binary_majority(csv: Path, cfg: BinaryTaskConfig) -> dict:
    """Per-week F1/accuracy and per-user AUROC/AUPRC for the always-majority baseline."""
    df = pd.read_csv(csv)
    gt_str = df["ground_truth"].astype(str)
    counts = gt_str.value_counts().to_dict()
    majority = max(counts, key=counts.get)

    # Per-week F1 / accuracy of the constant-majority predictor.
    y_true = (gt_str == cfg.positive).astype(int).to_numpy()
    y_pred = np.full_like(y_true, fill_value=int(majority == cfg.positive))
    f1_pos = float(f1_score(y_true, y_pred, zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    week_prevalence = float(y_true.mean())

    # Per-user metrics. A constant scorer ranks every user identically, so
    #   AUROC = 0.5 by definition,
    #   AUPRC = positive-class prevalence at the user level.
    # Following the paper convention, AUPRC is reported with the minority
    # class as positive, which is the conservative reporting choice for
    # imbalanced binary tasks (and the convention that reproduces the
    # published numbers in tables 4 / 5).
    per_user = df.groupby("user_id")["ground_truth"].first().astype(str)
    user_prevalence_eval_pos = float((per_user == cfg.positive).mean())
    auprc_minority = float(min(user_prevalence_eval_pos, 1.0 - user_prevalence_eval_pos))
    # Absolute counts at the user level — the AUPRC convention (minority) is
    # what reviewers will read; reporting n_positive in that convention so the
    # number matches the AUPRC denominator one-to-one.
    n_users = int(per_user.size)
    n_positive_minority = int(round(n_users * auprc_minority))

    return {
        "task": cfg.display,
        "type": "binary",
        "eval_positive_label": cfg.positive,
        "majority_label": majority,
        "n_weeks": int(len(df)),
        "n_users": n_users,
        "n_positive_users": n_positive_minority,
        "week_positive_prevalence": week_prevalence,
        "user_positive_prevalence_minority": auprc_minority,
        "per_week": {
            "f1_positive": f1_pos * 100,
            "macro_f1": f1_macro * 100,
            "accuracy": acc * 100,
        },
        "per_user": {"auroc": 50.0, "auprc": auprc_minority * 100},
    }


def compute_ordinal_majority(csv: Path, cfg: OrdinalTaskConfig) -> dict:
    """Spearman rho and QWK are degenerate for a constant predictor; report 0 plus MAE."""
    df = pd.read_csv(csv)
    gt = df["ground_truth"].astype(str)

    level_to_idx = {label: idx for idx, label in enumerate(cfg.levels)}
    valid_mask = gt.isin(level_to_idx)
    gt = gt[valid_mask]
    gt_idx = gt.map(level_to_idx).astype(int).to_numpy()

    counts = pd.Series(gt).value_counts().to_dict()
    majority = max(counts, key=counts.get)
    majority_idx = level_to_idx[majority]
    pred_idx = np.full_like(gt_idx, fill_value=majority_idx)

    qwk = float(
        cohen_kappa_score(gt_idx, pred_idx, weights="quadratic", labels=list(range(len(cfg.levels))))
    )
    mae = float(np.abs(gt_idx - pred_idx).mean())

    # User-level count for transparency (used by the appendix N table).
    valid_df = df[valid_mask]
    per_user = valid_df.groupby("user_id")["ground_truth"].first()

    return {
        "task": cfg.display,
        "type": "ordinal",
        "n_levels": len(cfg.levels),
        "majority_label": majority,
        "n_weeks": int(valid_mask.sum()),
        "n_users": int(per_user.size),
        "per_week": {"spearman_r": 0.0, "qwk": qwk * 100, "mae": mae},
        "per_user": {"spearman_r": 0.0, "qwk": qwk * 100},
    }


# Output formatting


def format_markdown(rows: list[dict]) -> str:
    binary = [r for r in rows if r["type"] == "binary"]
    ordinal = [r for r in rows if r["type"] == "ordinal"]

    def _fmt(val: float, digits: int = 2) -> str:
        return f"{val:.{digits}f}"

    out = ["# Majority-class baselines\n"]
    out.append(
        "Always-predict-majority on the held-out test split. AUROC is 0.5 by\n"
        "definition for any constant scorer; AUPRC reduces to the per-user\n"
        "positive-class prevalence. Spearman~ρ and QWK are degenerate (≈ 0) for\n"
        "a constant ordinal predictor.\n"
    )

    out.append("\n## Binary (per-week macro-F1 / accuracy; per-user AUROC / AUPRC)\n")
    out.append(
        "| Task | n_wk | n_users | n_pos | minority-prev | F1$_{+}$ | macro-F1 | Acc | "
        "AUROC | AUPRC |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in binary:
        out.append(
            f"| {r['task']} | {r['n_weeks']:,} | {r['n_users']:,} | "
            f"{r['n_positive_users']:,} | "
            f"{r['user_positive_prevalence_minority']:.3f} | "
            f"{_fmt(r['per_week']['f1_positive'])} | {_fmt(r['per_week']['macro_f1'])} | "
            f"{_fmt(r['per_week']['accuracy'])} | "
            f"{_fmt(r['per_user']['auroc'])} | {_fmt(r['per_user']['auprc'])} |"
        )

    out.append("\n## Ordinal (per-week QWK / MAE; per-user QWK)\n")
    out.append("| Task | n_wk | n_users | majority | QWK | MAE |")
    out.append("|---|---:|---:|---|---:|---:|")
    for r in ordinal:
        out.append(
            f"| {r['task']} | {r['n_weeks']:,} | {r['n_users']:,} | "
            f"{r['majority_label']} | "
            f"{_fmt(r['per_week']['qwk'])} | {_fmt(r['per_week']['mae'], 3)} |"
        )

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
            "Directory to write majority_baselines.{json,md} into. Defaults to the "
            "directory of this script so the snapshot stays version-controlled "
            "alongside it (results/ is gitignored)."
        ),
    )
    args = parser.parse_args()

    rows: list[dict] = []
    search_roots = [args.predictions_dir, args.secondary_dir]

    for task_name, cfg in BINARY_TASKS.items():
        csv = next(
            (c for root in search_roots for c in [_find_predictions_csv(root / task_name)] if c),
            None,
        )
        if csv is None:
            print(f"[skip] {cfg.display}: no predictions.csv under any search root")
            continue
        rows.append(compute_binary_majority(csv, cfg))

    for task_name, cfg in ORDINAL_TASKS.items():
        csv = next(
            (c for root in search_roots for c in [_find_predictions_csv(root / task_name)] if c),
            None,
        )
        if csv is None:
            print(f"[skip] {cfg.display}: no predictions.csv under any search root")
            continue
        rows.append(compute_ordinal_majority(csv, cfg))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "majority_baselines.json"
    md_path = args.output_dir / "majority_baselines.md"

    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    md_path.write_text(format_markdown(rows))

    print(format_markdown(rows))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
