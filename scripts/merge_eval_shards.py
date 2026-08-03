#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Recombine sharded WESAD or MIMIC runs into one result.

A shard's metrics.json describes only its slice, so the dataset-level numbers
have to be recomputed over the union rather than averaged: F1 and accuracy are
not linear in the shards, and per-shard standard errors are computed from
subsets and cannot be pooled by averaging either.

    python scripts/merge_eval_shards.py --task mimic_mortality_truefalse

Merges the newest complete shard set (same shard count, same model) under
results/<task>/<model>/ and writes results/<task>/<model>/<stamp>_merged/.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.tasks.mimic_mortality import MIMICMortalityTask  # noqa: E402
from evaluation.tasks.wesad_state import (  # noqa: E402
    WESAD_STATE_VALUES,
    WESADBinaryTask,
    WESADStateTask,
)

_SHARD_RE = re.compile(r"^(?P<stamp>\d{8}_\d{6})_shard(?P<idx>\d+)of(?P<total>\d+)$")


def _task_for(name: str):
    """Rebuild the task that produced ``results/<name>/`` so metrics match.

    Metrics are recomputed, not merged, so the task here must be the same one
    the shards ran under — a mismatch silently rescores against the wrong label
    space and yields parse_rate 0 rather than an error.
    """
    if name == "wesad_state":
        return WESADStateTask()
    if name == "mimic_mortality":
        return MIMICMortalityTask("native")
    if name == "mimic_mortality_truefalse":
        return MIMICMortalityTask("truefalse")
    state = name.removeprefix("wesad_")
    if state in WESAD_STATE_VALUES:
        return WESADBinaryTask(state)
    raise SystemExit(f"Unknown task {name!r}. Expected one of: {', '.join(TASK_NAMES)}")


TASK_NAMES = (
    ["wesad_state"]
    + [f"wesad_{s}" for s in WESAD_STATE_VALUES]
    + ["mimic_mortality", "mimic_mortality_truefalse"]
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sharded WESAD eval results and recompute metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", required=True, choices=TASK_NAMES)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--model", default=None,
        help="Model subdirectory. Defaults to the only one present.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    task_dir = Path(args.results_dir) / args.task
    if not task_dir.is_dir():
        raise SystemExit(f"No results for task {args.task} under {task_dir}")

    models = sorted(p for p in task_dir.iterdir() if p.is_dir())
    if args.model:
        models = [task_dir / args.model]
    if len(models) != 1:
        raise SystemExit(
            f"Found {len(models)} model dirs under {task_dir}; pass --model. "
            f"({', '.join(p.name for p in models)})"
        )
    model_dir = models[0]

    # Group shard dirs by (stamp, total) — one launch of N shards shares both.
    groups: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for path in model_dir.iterdir():
        m = _SHARD_RE.match(path.name)
        if m and (path / "predictions.csv").exists():
            groups[(m["stamp"], int(m["total"]))].append(path)

    complete = {k: v for k, v in groups.items() if len(v) == k[1]}
    if not complete:
        partial = {f"{k[0]} ({len(v)}/{k[1]})" for k, v in groups.items()}
        raise SystemExit(
            "No complete shard set found under "
            f"{model_dir}. Incomplete: {', '.join(sorted(partial)) or 'none'}"
        )

    stamp, total = max(complete)
    shards = sorted(complete[(stamp, total)])
    print(f"Merging {total} shards from {stamp} under {model_dir}")

    rows: list[dict] = []
    for shard in shards:
        with open(shard / "predictions.csv", newline="") as f:
            shard_rows = list(csv.DictReader(f))
        print(f"  {shard.name}: {len(shard_rows)} rows")
        rows.extend(shard_rows)

    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(
            f"Duplicate sample_ids across shards ({len(ids) - len(set(ids))} dupes) "
            "— the shards likely came from different launches."
        )

    task = _task_for(args.task)
    metrics = task.aggregate_metrics(
        [r["ground_truth"] for r in rows], [r["prediction"] for r in rows]
    )

    out_dir = model_dir / f"{stamp}_merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nMerged {len(rows)} predictions -> {out_dir}")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        elif not isinstance(val, dict):
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
