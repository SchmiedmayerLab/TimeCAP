#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Paper-style metrics for rationale-model test predictions (train_rationale.py output).

Per label, mirroring scripts/paper_results/model_metrics.py of the TimeCap paper:

* per-week macro-F1 and accuracy on the parsed <answer>; unparseable answers count as
  negative (abstention is penalised, nothing is dropped)
* per-user AUROC and AUPRC (AUPRC for the user-level minority class)

The paper's per-user score is the mean of hard predictions over each user's weeks. The
rationale test split holds one week per user, so that score is 0/1 and its AUROC collapses
to balanced accuracy. The primary AUROC/AUPRC here therefore use the model's continuous
score, log p(yes) - log p(no) at its own <answer> tag (mean per user); the hard-prediction
version is reported alongside as ``auroc_hard``.

Several runs can be compared; a run given several prediction files (e.g. seeds) reports
mean ± sd across them, a single file reports a bootstrap SE over users.

    python scripts/rationale_metrics.py \\
        --run "SFT=runs/rationale/seed7/.../rationale_grpo/results/baseline_test_predictions.jsonl" \\
        --run "GRPO-117k=runs/rationale_v2/.../rationale_grpo/results/test_predictions.jsonl" \\
        --out reports/rationale_metrics.md
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

DISPLAY = {"cardiovascular_disease": "CVD", "Hypertension": "Hypertension", "sleep_diagnosis1": "Sleep disorder",
           "Diabetes": "Diabetes", "CAD": "CAD", "Cerebrovascular Disease": "Stroke",
           "Atrial fibrillation (Afib)": "AFib", "PH": "PH"}
METRICS = ["macro_f1", "accuracy", "auroc", "auprc", "auroc_hard", "parse_rate"]


def load(path: str) -> pd.DataFrame:
    rows = [json.loads(line) for line in open(path)]
    df = pd.DataFrame(rows)
    # sample_id is "<label>|mhc_weekly:<user>:<week>" for GRPO rows
    df["user_id"] = df["sample_id"].str.split("|").str[-1].str.split(":").str[1]
    df["y"] = (df["gold"] == "yes").astype(int)
    df["yhat"] = (df["pred"] == "yes").astype(int)  # None / "no" -> 0
    return df


def label_metrics(df: pd.DataFrame) -> dict:
    out = {"n_users": df["user_id"].nunique(), "n_pos": int(df.groupby("user_id")["y"].first().sum()),
           "parse_rate": float(df["pred"].notna().mean()),
           "macro_f1": float(f1_score(df["y"], df["yhat"], labels=[0, 1], average="macro", zero_division=0)),
           "accuracy": float(accuracy_score(df["y"], df["yhat"]))}
    user = df.groupby("user_id").agg(y=("y", "first"), score=("score", "mean"), hard=("yhat", "mean"))
    if user["y"].nunique() < 2:
        return {**out, "auroc": np.nan, "auprc": np.nan, "auroc_hard": np.nan}
    minority_pos = user["y"].mean() <= 0.5
    y_min = user["y"] if minority_pos else 1 - user["y"]
    s_min = user["score"] if minority_pos else -user["score"]
    return {**out, "auroc": float(roc_auc_score(user["y"], user["score"])),
            "auprc": float(average_precision_score(y_min, s_min)),
            "auroc_hard": float(roc_auc_score(user["y"], user["hard"]))}


def all_metrics(df: pd.DataFrame) -> dict:
    per = {label: label_metrics(g) for label, g in df.groupby("label")}
    macro = {m: float(np.nanmean([v[m] for v in per.values()])) for m in METRICS}
    return {"per_label": per, "macro": macro}


def bootstrap_se(df: pd.DataFrame, n: int, seed: int) -> dict:
    """SE of every metric from resampling users with replacement (within label)."""
    rng = np.random.default_rng(seed)
    groups = {label: {u: g for u, g in lab.groupby("user_id")} for label, lab in df.groupby("label")}
    draws = []
    for _ in range(n):
        parts = []
        for label, users in groups.items():
            ids = list(users)
            for k, u in enumerate(rng.choice(len(ids), len(ids))):
                parts.append(users[ids[u]].assign(user_id=f"{ids[u]}#{k}"))
        draws.append(all_metrics(pd.concat(parts)))
    keys = list(groups)
    se = {"macro": {m: float(np.nanstd([d["macro"][m] for d in draws])) for m in METRICS},
          "per_label": {lab: {m: float(np.nanstd([d["per_label"][lab][m] for d in draws])) for m in METRICS}
                        for lab in keys}}
    return se


def summarize(name: str, paths: list[str], n_boot: int) -> dict:
    runs = [all_metrics(load(p)) for p in paths]
    if len(runs) > 1:  # mean ± sd across seeds
        mean = {"macro": {m: float(np.nanmean([r["macro"][m] for r in runs])) for m in METRICS},
                "per_label": {lab: {m: float(np.nanmean([r["per_label"][lab][m] for r in runs])) for m in METRICS}
                              for lab in runs[0]["per_label"]}}
        spread = {"macro": {m: float(np.nanstd([r["macro"][m] for r in runs], ddof=1)) for m in METRICS},
                  "per_label": {lab: {m: float(np.nanstd([r["per_label"][lab][m] for r in runs], ddof=1))
                                      for m in METRICS} for lab in runs[0]["per_label"]}}
        kind = f"sd over {len(runs)} runs"
    else:
        mean = runs[0]
        spread = bootstrap_se(load(paths[0]), n_boot, 0) if n_boot else None
        kind = f"bootstrap SE ({n_boot} user resamples)" if n_boot else "single run"
    counts = {lab: (v["n_users"], v["n_pos"]) for lab, v in runs[0]["per_label"].items()}
    return {"name": name, "files": paths, "mean": mean, "spread": spread, "spread_kind": kind, "counts": counts}


def fmt(m: float, s: float | None) -> str:
    return f"{m:.3f}" if s is None or np.isnan(s) else f"{m:.3f} ± {s:.3f}"


def markdown(results: list[dict]) -> str:
    lines = ["# Rationale model — test metrics", "",
             "macro-F1 / accuracy per week (unparsed = negative); AUROC / AUPRC per user from the model's "
             "yes-vs-no log-odds (AUPRC for the minority class); `AUROC hard` uses hard predictions as in the "
             "paper (one week per user, so it equals balanced accuracy).", ""]
    for r in results:
        lines.append(f"- **{r['name']}**: {r['spread_kind']}; {', '.join(r['files'])}")
    lines += ["", "## Macro over labels", "", "| Run | macro-F1 | Accuracy | AUROC | AUPRC | AUROC hard | parse rate |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        s = r["spread"]["macro"] if r["spread"] else {}
        lines.append(f"| {r['name']} | " + " | ".join(fmt(r["mean"]["macro"][m], s.get(m)) for m in METRICS) + " |")
    for m, title in (("auroc", "AUROC"), ("auprc", "AUPRC"), ("macro_f1", "macro-F1"), ("accuracy", "Accuracy")):
        labels = list(results[0]["counts"])
        lines += ["", f"## {title} per label", "",
                  "| Label | users (pos) | " + " | ".join(r["name"] for r in results) + " |",
                  "|---|---|" + "---|" * len(results)]
        for lab in labels:
            n, p = results[0]["counts"][lab]
            cells = [fmt(r["mean"]["per_label"][lab][m], (r["spread"]["per_label"][lab][m] if r["spread"] else None))
                     for r in results]
            lines.append(f"| {DISPLAY.get(lab, lab)} | {n} ({p}) | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", action="append", required=True,
                   help="NAME=path[,path...]; globs allowed; several files = mean ± sd across them.")
    p.add_argument("--bootstrap", type=int, default=1000, help="User resamples for single-file runs (0 = off).")
    p.add_argument("--out", default=None, help="Markdown report path; a .json with the numbers is written next to it.")
    args = p.parse_args()

    results = []
    for spec in args.run:
        name, paths = spec.split("=", 1)
        files = sorted(f for pat in paths.split(",") for f in glob.glob(pat))
        if not files:
            raise SystemExit(f"no files for {name}: {paths}")
        results.append(summarize(name, files, args.bootstrap))
    report = markdown(results)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report)
        Path(args.out).with_suffix(".json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
