#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Join structured rationales with their weekly windows into one arrow file for SFT.

Every rationale becomes one row carrying the 19x168 week, the participant's demographics,
the target label and the rationale text. Rationale users all come from the benchmark's
train split, so the SFT train / validation / test split is drawn by user here; users with
a positive for a label in ``--pin_train`` always land in train.

Example::

    python scripts/export_rationale_sft.py --rationales weekly_rationales_sft_v4.jsonl \\
        --weekly_dir data/sharable_weekly --demographics demographics.parquet \\
        --out rationale_sft_v4.arrow --pin_train "Cerebrovascular Disease"
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re

import pandas as pd
import pyarrow as pa


def load_weeks(weekly_dir: str, row_ids: set[str]) -> dict[str, dict]:
    weeks = {}
    cols = ["row_id", "user_id", "date", "values", "channel_names", "display_names", "units"]
    for path in sorted(glob.glob(os.path.join(weekly_dir, "**", "*.arrow"), recursive=True)):
        with pa.memory_map(path, "r") as src:
            table = pa.ipc.open_file(src).read_all().select(cols)
        for row in table.filter(pa.compute.is_in(table["row_id"], pa.array(list(row_ids)))).to_pylist():
            weeks[row["row_id"]] = row
    return weeks


def demographics(path: str) -> dict[str, dict]:
    """user_id -> {age, bmi, sex}; -1 / non-positive values mean unknown."""
    d = pd.read_parquet(path).drop_duplicates("user_id").set_index("user_id")
    return {u: {"age": float(r.age) if r.age > 0 else None,
                "bmi": float(r.BMI_values) if r.BMI_values > 0 else None,
                "sex": {1: "male", 0: "female"}.get(int(r.BiologicalSex))}
            for u, r in d.iterrows()}


def write_rows(rows: list[dict], out: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    table = pa.Table.from_pylist(rows)
    with pa.ipc.new_file(out, table.schema) as writer:
        writer.write_table(table)


def summarize(rows: list[dict]) -> None:
    counts = collections.Counter((r["split"], r["label"], r["answer"]) for r in rows)
    for split in ("train", "validation", "test"):
        n_users = len({r["user_id"] for r in rows if r["split"] == split})
        print(f"{split}: {sum(v for k, v in counts.items() if k[0] == split)} rows, {n_users} users")
        for (s, label, answer), n in sorted(counts.items()):
            if s == split:
                print(f"  {label:30s} {answer:3s} {n}")


def split_users(records: list[dict], pin_train: set[str], frac: float, seed: int) -> dict[str, str]:
    pinned = {r["user_id"] for r in records if r["label"] in pin_train and r["ground_truth"] == "positive"}
    users = sorted({r["user_id"] for r in records} - pinned)
    random.Random(seed).shuffle(users)
    n = round(len(users) * frac)
    bucket = {u: "train" for u in pinned}
    bucket.update({u: "validation" for u in users[:n]})
    bucket.update({u: "test" for u in users[n:2 * n]})
    bucket.update({u: "train" for u in users[2 * n:]})
    return bucket


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rationales", required=True)
    p.add_argument("--weekly_dir", required=True)
    p.add_argument("--demographics", required=True,
                   help="Parquet with user_id, age, BMI_values, BiologicalSex (-1 = unknown).")
    p.add_argument("--out", required=True)
    p.add_argument("--pin_train", nargs="*", default=[], help="Labels whose positive users stay in train.")
    p.add_argument("--holdout_frac", type=float, default=0.1, help="Fraction of users for each of validation and test.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    records = [json.loads(line) for line in open(args.rationales)]
    weeks = load_weeks(args.weekly_dir, {r["row_id"] for r in records})
    missing = {r["row_id"] for r in records} - set(weeks)
    assert not missing, f"{len(missing)} weeks not found, e.g. {sorted(missing)[:3]}"

    demo = demographics(args.demographics)
    bucket = split_users(records, set(args.pin_train), args.holdout_frac, args.seed)

    rows = []
    for r in records:
        week = weeks[r["row_id"]]
        answer = re.findall(r"<answer>(yes|no)</answer>", r["rationale"])[-1]
        assert answer == ("yes" if r["ground_truth"] == "positive" else "no"), r["id"]
        rows.append({
            "sample_id": r["id"], "row_id": r["row_id"], "user_id": r["user_id"], "date": week["date"],
            "label": r["label"], "ground_truth": r["ground_truth"], "answer": answer,
            "rationale": r["rationale"], "split": bucket[r["user_id"]], **demo[r["user_id"]],
            "values": week["values"], "channel_names": week["channel_names"],
            "display_names": week["display_names"], "units": week["units"],
        })

    write_rows(rows, args.out)
    summarize(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
