#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Recompute paper metrics from prediction CSVs.

For every (task, model run) under the input directories this writes:

* Per-week binary: macro-F1 and accuracy.
* Per-user binary: AUROC and AUPRC over the per-user mean of binary predictions.
* Per-week + per-user ordinal: Spearman~rho, quadratic weighted kappa, MAE.

Binary unknowns (predictions the parser couldn't map to either label) are
counted as negative-class predictions (0). This penalizes abstention rather
than dropping unparseable rows, and is applied uniformly at both the per-week
and per-user level. Ordinal unknowns are still dropped — there is no
principled "wrong" rank to assign — so the parse_rate column should be read
alongside the ordinal numbers.

Macro-F1 (mean of class-0 F1 and class-1 F1) is reported instead of binary F1
because the binary form is asymmetric in the label encoding: predicting only
the majority class scores ~0 when the positive label is the minority but
2*p/(p+1) when the positive label is the majority. Macro-F1 removes that
artefact so the per-week table can be read independently of which class each
task happens to encode as positive.

Outputs land next to the script (results/ is gitignored) so the metric
snapshot is version-controlled alongside the code that produced it.

Run::

    python scripts/paper_results/model_metrics.py \\
        --predictions_dir ~/Downloads/results \\
        --secondary_dir "~/Downloads/results 2"
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


# Task and model configuration


@dataclass(frozen=True)
class BinaryTaskConfig:
    display: str
    positive: str  # canonical positive label string


@dataclass(frozen=True)
class OrdinalTaskConfig:
    display: str
    levels: tuple[str, ...]  # ordered low -> high


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


# Eval output directory name -> display label used in tables.
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "opentslm-latest":             "OpenTSLM-1B",
    "opentslm-joint_bs2_final_1b": "OpenTSLM-Cap-1B",
    "opentslm-metabolic_3b_final": "OpenTSLM-3B",
    "opentslm-joint_3b_final":     "OpenTSLM-Cap-3B",
}


# Computation


def _safe_pct(x: float) -> float | None:
    """Multiply by 100, mapping NaN to None for clean JSON output."""
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else x * 100


def compute_binary(df: pd.DataFrame, cfg: BinaryTaskConfig) -> dict:
    """Per-week macro-F1/accuracy and per-user AUROC/AUPRC; unknowns count as negative."""
    n_total = int(len(df))
    n_parsed = int((df["prediction"] != "unknown").sum())

    out: dict = {
        "n_weeks_total": n_total,
        "n_weeks_parsed": n_parsed,
        "parse_rate": (n_parsed / n_total) if n_total else 0.0,
    }
    if n_total == 0:
        out.update(per_week=None, per_user=None, n_users=0)
        return out

    df = df.copy()
    y_true_wk = (df["ground_truth"].astype(str) == cfg.positive).astype(int).to_numpy()
    y_pred_wk = (df["prediction"].astype(str) == cfg.positive).astype(int).to_numpy()

    macro_f1 = f1_score(y_true_wk, y_pred_wk, labels=[0, 1], average="macro", zero_division=0)
    acc = accuracy_score(y_true_wk, y_pred_wk)

    df = df.assign(_pred=y_pred_wk, _gt=y_true_wk)
    per_user = df.groupby("user_id").agg(score=("_pred", "mean"), gt=("_gt", "first"))

    if per_user["gt"].nunique() < 2:
        auroc = float("nan")
        auprc = float(per_user["gt"].mean()) if per_user["gt"].any() else 0.0
        auprc_label = cfg.positive
    else:
        auroc = float(roc_auc_score(per_user["gt"], per_user["score"]))
        # Pick the user-level minority class for AUPRC; if positive is already
        # the minority we keep it, otherwise we flip both score and ground
        # truth so AUPRC reports "how well the model identifies the rarer
        # class" — matches the paper convention.
        if per_user["gt"].mean() <= 0.5:
            auprc_gt, auprc_score = per_user["gt"], per_user["score"]
            auprc_label = cfg.positive
        else:
            auprc_gt = 1 - per_user["gt"]
            auprc_score = 1 - per_user["score"]
            # Find the negation of cfg.positive among the actual GT strings.
            other = sorted(df["ground_truth"].astype(str).unique())
            auprc_label = next((s for s in other if s != cfg.positive), cfg.positive)
        auprc = float(average_precision_score(auprc_gt, auprc_score))

    out["n_users"] = int(len(per_user))
    out["per_week"] = {"macro_f1": _safe_pct(macro_f1), "accuracy": _safe_pct(acc)}
    out["per_user"] = {
        "auroc": _safe_pct(auroc),
        "auprc": _safe_pct(auprc),
        "auprc_positive_label": auprc_label,
    }
    return out


def compute_ordinal(df: pd.DataFrame, cfg: OrdinalTaskConfig) -> dict:
    """Per-week and per-user Spearman rho, QWK, and MAE; unknowns dropped."""
    level_to_idx = {label: idx for idx, label in enumerate(cfg.levels)}
    parsed = df[df["prediction"] != "unknown"].copy()
    parsed = parsed[
        parsed["prediction"].isin(level_to_idx)
        & parsed["ground_truth"].isin(level_to_idx)
    ]
    n_total = int(len(df))
    n_parsed = int(len(parsed))

    out: dict = {
        "n_weeks_total": n_total,
        "n_weeks_parsed": n_parsed,
        "parse_rate": (n_parsed / n_total) if n_total else 0.0,
    }
    if n_parsed == 0:
        out.update(per_week=None, per_user=None, n_users=0)
        return out

    gt_idx = parsed["ground_truth"].map(level_to_idx).astype(int).to_numpy()
    pred_idx = parsed["prediction"].map(level_to_idx).astype(int).to_numpy()
    label_range = list(range(len(cfg.levels)))

    rho_wk = (
        float(spearmanr(gt_idx, pred_idx)[0])
        if (np.std(gt_idx) > 0 and np.std(pred_idx) > 0)
        else float("nan")
    )
    qwk_wk = float(cohen_kappa_score(gt_idx, pred_idx, weights="quadratic", labels=label_range))
    mae_wk = float(np.abs(gt_idx - pred_idx).mean())

    parsed = parsed.assign(_gt=gt_idx, _pred=pred_idx)
    per_user = parsed.groupby("user_id").agg(score=("_pred", "mean"), gt=("_gt", "first"))
    pred_rounded = per_user["score"].round().clip(0, len(cfg.levels) - 1).astype(int).to_numpy()
    gt_user = per_user["gt"].to_numpy()

    rho_user = (
        float(spearmanr(gt_user, per_user["score"])[0])
        if (np.std(gt_user) > 0 and np.std(per_user["score"]) > 0)
        else float("nan")
    )
    qwk_user = (
        float(cohen_kappa_score(gt_user, pred_rounded, weights="quadratic", labels=label_range))
        if np.std(gt_user) > 0
        else 0.0
    )
    mae_user = float(np.abs(gt_user - pred_rounded).mean())

    out["n_users"] = int(len(per_user))
    out["per_week"] = {"spearman_r": _safe_pct(rho_wk), "qwk": _safe_pct(qwk_wk), "mae": mae_wk}
    out["per_user"] = {"spearman_r": _safe_pct(rho_user), "qwk": _safe_pct(qwk_user), "mae": mae_user}
    return out


# Discovery


def discover_runs(roots: list[Path]) -> dict[tuple[str, str], tuple[Path, Path]]:
    """Map (task_dir_name, model_dir_name) -> (longest predictions.csv path, root)."""
    found: dict[tuple[str, str], tuple[Path, Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for pred_csv in root.rglob("predictions.csv"):
            rel = pred_csv.relative_to(root).parts
            if len(rel) < 2:
                continue
            task_dir, model_dir = rel[0], rel[1]
            key = (task_dir, model_dir)
            existing = found.get(key)
            if existing is None or len(pd.read_csv(pred_csv)) > len(pd.read_csv(existing[0])):
                found[key] = (pred_csv, root)
    return found


# Output formatting


def format_markdown(records: list[dict]) -> str:
    binary = [r for r in records if r["type"] == "binary"]
    ordinal = [r for r in records if r["type"] == "ordinal"]

    def _fmt(val: float | None, digits: int = 2) -> str:
        if val is None:
            return "—"
        return f"{val:.{digits}f}"

    lines: list[str] = ["# Model metrics (recomputed from prediction CSVs)\n"]
    lines.append(
        "Per-week macro-F1 / accuracy for binary tasks (label-symmetric: a "
        "majority-class predictor scores ~accuracy/(1+accuracy) on the dominant "
        "class and 0 on the other, so macro-F1 is bounded by the imbalance).\n"
        "Per-user AUROC / AUPRC use the per-user mean of binary predictions as "
        "the score and the user's static label as ground truth.\n"
        "Binary unknowns count as a negative-class prediction (0) at both "
        "granularities — abstention is penalized, not dropped.\n"
    )

    lines.append("\n## Binary tasks\n")
    lines.append(
        "| Task | Model | n_users | parse_rate | macro-F1 | Acc | AUROC | AUPRC |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in binary:
        pw = r["per_week"] or {}
        pu = r["per_user"] or {}
        lines.append(
            f"| {r['task']} | {r['model']} | {r['n_users']} | "
            f"{r['parse_rate']*100:.1f}\\% | "
            f"{_fmt(pw.get('macro_f1'))} | {_fmt(pw.get('accuracy'))} | "
            f"{_fmt(pu.get('auroc'))} | {_fmt(pu.get('auprc'))} |"
        )

    lines.append("\n## Ordinal tasks\n")
    lines.append(
        "| Task | Model | n_users | parse_rate | wk-rho | wk-QWK | "
        "user-rho | user-QWK | user-MAE |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in ordinal:
        pw = r["per_week"] or {}
        pu = r["per_user"] or {}
        lines.append(
            f"| {r['task']} | {r['model']} | {r['n_users']} | "
            f"{r['parse_rate']*100:.1f}\\% | "
            f"{_fmt(pw.get('spearman_r'))} | {_fmt(pw.get('qwk'))} | "
            f"{_fmt(pu.get('spearman_r'))} | {_fmt(pu.get('qwk'))} | "
            f"{_fmt(pu.get('mae'), 3)} |"
        )

    return "\n".join(lines) + "\n"


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
            "Directory to write model_metrics.{json,md} into. Defaults to the "
            "directory of this script so the snapshot stays version-controlled "
            "alongside it (results/ is gitignored)."
        ),
    )
    args = parser.parse_args()

    runs = discover_runs([args.predictions_dir, args.secondary_dir])
    records: list[dict] = []

    for (task_dir, model_dir), (csv_path, search_root) in sorted(runs.items()):
        cfg_b = BINARY_TASKS.get(task_dir)
        cfg_o = ORDINAL_TASKS.get(task_dir)
        if cfg_b is None and cfg_o is None:
            print(f"[skip] unknown task dir {task_dir!r}")
            continue
        model_label = MODEL_DISPLAY_NAMES.get(model_dir, model_dir)
        df = pd.read_csv(csv_path)
        # Keep the path stable across machines: store it relative to whichever
        # search root it was discovered under, so the snapshot doesn't leak
        # local usernames or churn on diff between developers.
        csv_rel = str(csv_path.relative_to(search_root))

        if cfg_b is not None:
            metrics = compute_binary(df, cfg_b)
            metrics.update(task=cfg_b.display, model=model_label, type="binary",
                           task_dir=task_dir, model_dir=model_dir, csv=csv_rel)
        else:
            metrics = compute_ordinal(df, cfg_o)
            metrics.update(task=cfg_o.display, model=model_label, type="ordinal",
                           task_dir=task_dir, model_dir=model_dir, csv=csv_rel)
        records.append(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "model_metrics.json"
    md_path = args.output_dir / "model_metrics.md"

    json_path.write_text(json.dumps(records, indent=2) + "\n")
    md_path.write_text(format_markdown(records))

    print(format_markdown(records))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
