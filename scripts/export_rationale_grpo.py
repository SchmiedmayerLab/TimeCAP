#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Export label-only weekly prompts for GRPO in the rationale arrow format.

GRPO only needs the label for its reward, so rows carry no rationale. Splits follow
MHC-benchmark. Train prompts come from train users, in one of two modes:

  balanced  ``--train_per_class`` weeks per label and class, cycling through users (one
            week each before anyone's second week, so rare positives contribute several)
  all_pos   every positive (label, week) pair once, plus as many negative weeks per label,
            drawn person-first so every negative user contributes before anyone repeats

Validation and test take one week per user, every user up to a per-class cap, or are
copied unchanged from an earlier export with ``--eval_from`` so results stay comparable.

``--test_only --test_max_per_user 16 --test_per_class 1000000`` instead writes a test-only
file with every test user and up to 16 random weeks each, for per-user evaluation over
several weeks (as in the TimeCap paper).

Example::

    python scripts/export_rationale_grpo.py --weekly_dir ../data/sharable_weekly \\
        --labels ../data/weekly_labels_lookup_user.parquet --demographics demographics.parquet \\
        --split_file ../data/sharable_users_seed42_2026.json --out rationale_grpo_v1.arrow
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import numpy as np
import pandas as pd
import pyarrow as pa

from export_rationale_sft import demographics, load_weeks, summarize, write_rows

DEFAULT_LABELS = ["cardiovascular_disease", "Hypertension", "sleep_diagnosis1", "Diabetes", "CAD",
                  "Cerebrovascular Disease", "Atrial fibrillation (Afib)", "PH"]
HR_CHANNEL = "hk_watch:HKQuantityTypeIdentifierHeartRate"


def usable_weeks(weekly_dir: str, users: set[str], min_hr_days: int) -> dict[str, list[str]]:
    """user_id -> row_ids of weeks with heart rate on at least ``min_hr_days`` days.

    Only ids are kept so all users fit in memory; selected weeks are loaded afterwards.
    """
    weeks: dict[str, list[str]] = {}
    for path in sorted(glob.glob(os.path.join(weekly_dir, "**", "*.arrow"), recursive=True)):
        with pa.memory_map(path, "r") as src:
            table = pa.ipc.open_file(src).read_all().select(["row_id", "user_id", "values", "channel_names"])
        table = table.filter(pa.compute.is_in(table["user_id"], pa.array(list(users))))
        for batch in table.to_batches(max_chunksize=256):
            for row in batch.to_pylist():
                values = np.asarray(row["values"], dtype=np.float32)
                hr = values[row["channel_names"].index(HR_CHANNEL)]
                if values.shape == (19, 168) and np.isfinite(hr.reshape(7, 24)).any(axis=1).sum() >= min_hr_days:
                    weeks.setdefault(row["user_id"], []).append(row["row_id"])
        del table
    return weeks


def pick(weeks: dict[str, list[str]], users: list[str], want: int, max_per_user: int,
         rng: random.Random) -> list[str]:
    """Round-robin over shuffled users until ``want`` weeks or the per-user cap is hit."""
    users = sorted(users)
    rng.shuffle(users)
    pools = {u: rng.sample(weeks[u], len(weeks[u])) for u in users}
    picked = []
    for depth in range(max_per_user):
        for u in users:
            if len(picked) >= want:
                return picked
            if depth < len(pools[u]):
                picked.append(pools[u][depth])
    return picked


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weekly_dir", required=True)
    p.add_argument("--labels", required=True, help="Parquet with user_id + label columns (1 / 0 / -1).")
    p.add_argument("--demographics", required=True)
    p.add_argument("--split_file", required=True, help="MHC-benchmark sharable_users_seed42_2026.json.")
    p.add_argument("--out", required=True)
    p.add_argument("--targets", nargs="+", default=DEFAULT_LABELS)
    p.add_argument("--train_mode", choices=["balanced", "all_pos"], default="balanced")
    p.add_argument("--train_per_class", type=int, default=60)
    p.add_argument("--train_max_per_user", type=int, default=6)
    p.add_argument("--val_per_class", type=int, default=15)
    p.add_argument("--test_per_class", type=int, default=100)
    p.add_argument("--test_max_per_user", type=int, default=1)
    p.add_argument("--test_only", action="store_true", help="Export only the test split.")
    p.add_argument("--min_hr_days", type=int, default=5)
    p.add_argument("--eval_from", default=None, help="Copy validation/test rows from this arrow file.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)

    splits = json.load(open(args.split_file))
    labels = pd.read_parquet(args.labels).drop_duplicates("user_id").set_index("user_id")
    demo = demographics(args.demographics)
    use_splits = ("test",) if args.test_only else ("train",) if args.eval_from else ("train", "validation", "test")
    users = (set(labels.index) & set(demo)) & {u for s in use_splits for u in splits[s]}
    weeks = usable_weeks(args.weekly_dir, users, args.min_hr_days)
    print(f"{len(weeks)} users with usable weeks")

    plan = {"train": (args.train_per_class, args.train_max_per_user)}
    if not args.eval_from:
        plan.update(validation=(args.val_per_class, 1), test=(args.test_per_class, args.test_max_per_user))
    if args.test_only:
        plan = {"test": (args.test_per_class, args.test_max_per_user)}
    chosen = []  # (split, label, value, row_id)
    for split, (want, max_per_user) in plan.items():
        split_users = [u for u in splits[split] if u in weeks]
        for label in args.targets:
            pos = [u for u in split_users if labels.at[u, label] == 1]
            neg = [u for u in split_users if labels.at[u, label] == 0]
            if split == "train" and args.train_mode == "all_pos":
                pos_weeks = [r for u in pos for r in weeks[u]]
                chosen += [(split, label, 1, r) for r in pos_weeks]
                chosen += [(split, label, 0, r) for r in pick(weeks, neg, len(pos_weeks), 10**6, rng)]
                continue
            for value, cls_users in ((1, pos), (0, neg)):
                chosen += [(split, label, value, r) for r in pick(weeks, cls_users, want, max_per_user, rng)]
    loaded = load_weeks(args.weekly_dir, {c[3] for c in chosen})

    rows = []
    for split, label, value, row_id in chosen:
        week = loaded[row_id]
        rows.append({
            "sample_id": f"{label}|{row_id}", "row_id": row_id, "user_id": week["user_id"],
            "date": week["date"], "label": label, "ground_truth": "positive" if value else "negative",
            "answer": "yes" if value else "no", "rationale": None, "split": split, **demo[week["user_id"]],
            "values": week["values"], "channel_names": week["channel_names"],
            "display_names": week["display_names"], "units": week["units"],
        })
    if args.eval_from:
        with pa.memory_map(args.eval_from, "r") as src:
            table = pa.ipc.open_file(src).read_all()
        rows += table.filter(pa.compute.not_equal(table["split"], "train")).to_pylist()
    rng.shuffle(rows)
    write_rows(rows, args.out)
    summarize(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
