#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Convert MHC-benchmark XGBoost per-user test predictions to the rationale prediction format.

Keeps exactly the (label, user) rows of a reference rationale prediction file, so
``scripts/rationale_metrics.py`` scores XGBoost and the TSLM on identical rows. ``pred`` is
the harness's own hard prediction (``y_pred``), ``score`` its probability. With
``--per_user`` a multi-week reference is reduced to one row per (label, user), since the
XGBoost model predicts per user.

    python scripts/xgb_to_rationale_predictions.py \\
        --xgb_dir ../../MHC/MHC-benchmark/results/eval/rationale_compare/fx_clinical/predictions/fe_xgboost \\
        --reference reports/preds/grpo117k_step700.jsonl --out reports/preds/xgb_clinical.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--xgb_dir", required=True, help="predictions/<method> dir with <task>/test.parquet")
    p.add_argument("--reference", required=True, help="Rationale test_predictions.jsonl defining the rows.")
    p.add_argument("--out", required=True)
    p.add_argument("--per_user", action="store_true", help="One row per (label, user).")
    args = p.parse_args()

    ref = [json.loads(line) for line in open(args.reference)]
    if args.per_user:
        first = {}
        for r in ref:
            first.setdefault((r["label"], r["sample_id"].split("|")[-1].split(":")[1]), r)
        ref = list(first.values())
    tables = {}
    rows, missing, mismatch = [], 0, 0
    for r in ref:
        label = r["label"]
        if label not in tables:  # the harness sanitises task names for directories
            path = Path(args.xgb_dir) / re.sub(r"[^\w()-]+", "_", label) / "test.parquet"
            tables[label] = pd.read_parquet(path).set_index("uid")
        user = r["sample_id"].split("|")[-1].split(":")[1]
        if user not in tables[label].index:
            missing += 1
            continue
        x = tables[label].loc[user]
        gold = "yes" if int(x["y_true"]) == 1 else "no"
        mismatch += gold != r["gold"]
        rows.append({"idx": r["idx"], "sample_id": r["sample_id"], "label": label, "gold": r["gold"],
                     "pred": "yes" if int(x["y_pred"]) == 1 else "no", "score": float(x["y_proba"]),
                     "well_formed": True, "reward": None, "generated": "", "loss": None})
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)}/{len(ref)} rows matched; {missing} users without an XGBoost prediction; "
          f"{mismatch} ground-truth mismatches -> {args.out}")


if __name__ == "__main__":
    main()
