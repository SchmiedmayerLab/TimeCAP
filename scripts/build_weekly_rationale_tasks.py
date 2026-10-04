#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Build label-blind rationale-writing tasks from weekly MHC windows.

For each target label, picks weeks from positive and negative users outside the
validation/test split (one week per user first, round-robin after), computes the weekly
clinical findings, and writes

  <out>             tasks: row_id, label, rendered <findings> block, structured findings
  <out>.key.jsonl   ground truth per task, kept separate so the rationale writer never sees it

Example::

    python scripts/build_weekly_rationale_tasks.py \\
        --weekly_dir data/sharable_weekly --labels data/weekly_labels_lookup_user.parquet \\
        --split_file data/sharable_users_seed42_2026.json --out rationale_tasks.jsonl \\
        --plan cardiovascular_disease:75:40 Hypertension:70:40 sleep_diagnosis1:75:40
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical.data import CHANNEL_NAMES
from clinical.evidence import STRENGTH_ORDER, demographic_findings
from clinical.weekly_findings import _recording, weekly_findings

HR_IDX = CHANNEL_NAMES.index("hk_watch:HKQuantityTypeIdentifierHeartRate")
DISPLAY = {"cardiovascular_disease": "cardiovascular disease", "Hypertension": "hypertension",
           "sleep_diagnosis1": "a sleep disorder", "Diabetes": "diabetes",
           "Heart Failure or CHF": "heart failure (CHF)",
           "CAD": "coronary artery disease (CAD)", "Cerebrovascular Disease": "cerebrovascular disease (stroke)",
           "Atrial fibrillation (Afib)": "atrial fibrillation (AFib)", "PH": "pulmonary hypertension (PH)"}


def render(findings, label: str, sex: str | None = None) -> str:
    disp = DISPLAY.get(label, label.replace("_", " "))
    lines = [f"- {f.render(disp)}" for f in findings]
    if sex:
        lines.append(f"- Biological sex: {sex} (context)")
    return "<findings>\n" + "\n".join(lines) + "\n</findings>"


def load_weeks(weekly_dir: str, users: set[str]) -> dict[str, list[tuple[str, str, np.ndarray]]]:
    """user_id -> [(row_id, week_start, values 19x168)] for weeks with HR on >= 5 days."""
    weeks: dict[str, list] = {}
    for path in sorted(glob.glob(os.path.join(weekly_dir, "**", "*.arrow"), recursive=True)):
        with pa.memory_map(path, "r") as src:
            table = pa.ipc.open_file(src).read_all().select(["row_id", "user_id", "date", "values"])
        for row in table.filter(pa.compute.is_in(table["user_id"], pa.array(list(users)))).to_pylist():
            values = np.asarray(row["values"], dtype=np.float32)
            hr_days = np.isfinite(values[HR_IDX].reshape(7, 24)).any(axis=1).sum()
            if values.shape == (19, 168) and hr_days >= 5:
                weeks.setdefault(row["user_id"], []).append((row["row_id"], str(row["date"]), values))
    return weeks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weekly_dir", required=True)
    p.add_argument("--labels", required=True, help="Parquet with user_id + label columns (1 / 0 / -1).")
    p.add_argument("--split_file", required=True, help="Validation and test users are excluded.")
    p.add_argument("--plan", nargs="+", required=True, help="label:n_positive:n_negative")
    p.add_argument("--demographics", default=None,
                   help="Parquet with user_id, age, BMI_values, BiologicalSex (1 = male, -1 = unknown).")
    p.add_argument("--min_findings", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)

    split = json.loads(Path(args.split_file).read_text())
    held_out = set(split["validation"]) | set(split["test"])
    plan = [(lbl, int(pos), int(neg)) for lbl, pos, neg in (x.split(":") for x in args.plan)]
    labels = pd.read_parquet(args.labels).drop_duplicates("user_id").set_index("user_id")
    eligible = {u for u in labels.index if u not in held_out}
    demo = {}
    if args.demographics:
        d = pd.read_parquet(args.demographics).drop_duplicates("user_id").set_index("user_id")
        valid = lambda v: None if v is None or pd.isna(v) or v == -1 else float(v)
        demo = {u: (valid(r.age), valid(r.BMI_values), {1: "male", 0: "female"}.get(r.BiologicalSex))
                for u, r in d.iterrows()}
    if demo:
        eligible &= set(demo)
    weeks = load_weeks(args.weekly_dir, eligible)
    print(f"{len(weeks)} non-held-out users with usable weeks")

    tasks, keys = [], []
    for label, n_pos, n_neg in plan:
        for cls, value, want in (("positive", 1, n_pos), ("negative", 0, n_neg)):
            users = [u for u in weeks if labels.at[u, label] == value]
            rng.shuffle(users)
            for u in users:
                rng.shuffle(weeks[u])
            got, cursor = [], {u: 0 for u in users}
            while len(got) < want and any(cursor[u] < len(weeks[u]) for u in users):
                for u in users:
                    if len(got) >= want:
                        break
                    while cursor[u] < len(weeks[u]):
                        row_id, date, values = weeks[u][cursor[u]]
                        cursor[u] += 1
                        findings = weekly_findings(_recording(row_id, u, date, values), label)
                        if len(findings) >= args.min_findings:
                            age, bmi, _sex = demo.get(u, (None, None, None))
                            findings = sorted(findings + demographic_findings(age, bmi, label),
                                              key=lambda f: (STRENGTH_ORDER[f.strength], not f.supports))
                            got.append((row_id, u, date, findings))
                            break
            print(f"{label} {cls}: {len(got)} weeks from {len({g[1] for g in got})} of {len(users)} users")
            for row_id, u, date, findings in got:
                sex = demo.get(u, (None, None, None))[2]
                tasks.append({"row_id": row_id, "label": label, "findings": render(findings, label, sex),
                              "findings_struct": [vars(f) for f in findings]})
                keys.append({"row_id": row_id, "user_id": u, "week_start": date, "label": label,
                             "ground_truth": cls})

    order = list(range(len(tasks)))
    rng.shuffle(order)
    with open(args.out, "w") as ft, open(args.out.replace(".jsonl", "") + ".key.jsonl", "w") as fk:
        for i, j in enumerate(order):
            ft.write(json.dumps({"task": i, **tasks[j]}) + "\n")
            fk.write(json.dumps({"task": i, **keys[j]}) + "\n")
    print(f"{len(tasks)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
