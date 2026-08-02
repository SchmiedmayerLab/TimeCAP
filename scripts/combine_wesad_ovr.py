#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Combine four one-vs-rest WESAD runs into a 4-way prediction.

Each ``wesad_<state>`` run answers "is this window <state>?" with True/False —
a label space the checkpoint has actually seen, unlike the 4-way question. This
joins the four runs per window and resolves them into one state.

    python scripts/combine_wesad_ovr.py

Resolution is deliberately strict. Greedy decoding yields a hard True/False
with no confidence attached, so there is nothing principled to break ties with:

* exactly one True  -> that state
* two or more True  -> ``unknown`` (conflict)
* zero True         -> ``unknown`` (abstention)

Conflicts and abstentions are counted separately and reported, because they are
the interesting quantity: a model answering True to three of four questions is
telling you something different from one answering False to all four. Scoring
the True token's likelihood would resolve both, and is the forced-choice
protocol ``run_ablation_sweep.sh`` refers to — this script is the version that
needs no logprob access.

Metrics come from ``WESADStateTask.aggregate_metrics``, the same function the
native 4-way run uses, so the numbers are directly comparable to it.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.tasks.wesad_state import (  # noqa: E402
    WESAD_STATE_VALUES,
    WESADStateTask,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine one-vs-rest WESAD runs into a 4-way result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--model", default=None,
        help="Model subdir under results/wesad_<state>/. Defaults to the only one.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output dir. Defaults to results/wesad_ovr/<model>/<stamp>/.",
    )
    return parser.parse_args()


def _load_state(results_dir: Path, state: str, model: str | None) -> dict[str, dict]:
    """Newest run for one state, as {sample_id: row}. Prefers merged shard sets."""
    task_dir = results_dir / f"wesad_{state}"
    if not task_dir.is_dir():
        raise SystemExit(
            f"Missing {task_dir} — run run_wesad_eval.py --binary-label {state}"
        )
    models = [task_dir / model] if model else sorted(
        p for p in task_dir.iterdir() if p.is_dir()
    )
    if len(models) != 1:
        raise SystemExit(
            f"Found {len(models)} model dirs under {task_dir}; pass --model."
        )

    runs = [p for p in models[0].iterdir() if (p / "predictions.csv").exists()]
    if not runs:
        raise SystemExit(f"No predictions.csv under {models[0]}")
    # A merged shard set supersedes the individual shards it was built from.
    merged = [p for p in runs if p.name.endswith("_merged")]
    unsharded = [p for p in runs if "_shard" not in p.name and not p.name.endswith("_merged")]
    candidates = merged or unsharded or runs
    run = max(candidates, key=lambda p: p.name)

    with open(run / "predictions.csv", newline="") as f:
        rows = {r["sample_id"]: r for r in csv.DictReader(f)}
    print(f"  wesad_{state:<11} {run.name:<40} {len(rows):>5} windows")
    return rows


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)

    print("Loading one-vs-rest runs:")
    per_state = {s: _load_state(results_dir, s, args.model) for s in WESAD_STATE_VALUES}

    id_sets = [set(rows) for rows in per_state.values()]
    common = set.intersection(*id_sets)
    union = set.union(*id_sets)
    if not common:
        raise SystemExit("No sample_ids shared across all four runs.")
    if common != union:
        # Not fatal, but it means the four runs did not cover the same windows —
        # combining only the intersection keeps the 4-way result self-consistent.
        print(f"\nWARNING: {len(union - common)} windows missing from at least "
              f"one run; combining the {len(common)} present in all four.")

    ground_truths: list[str] = []
    predictions: list[str] = []
    combined_rows: list[dict] = []
    tally = collections.Counter()

    for sample_id in sorted(common):
        votes = [s for s in WESAD_STATE_VALUES if per_state[s][sample_id]["prediction"] == "True"]
        unparsed = [s for s in WESAD_STATE_VALUES
                    if per_state[s][sample_id]["prediction"] == "unknown"]

        if len(votes) == 1:
            prediction, outcome = votes[0], "resolved"
        elif len(votes) > 1:
            prediction, outcome = "unknown", "conflict"
        else:
            prediction, outcome = "unknown", "abstain"
        tally[outcome] += 1
        if unparsed:
            tally["had_unparsed_component"] += 1

        # Ground truth is identical across the four runs by construction (each
        # derives it from the same window), so recover the 4-way label from
        # whichever run says True for its own state.
        truth = next(
            (s for s in WESAD_STATE_VALUES
             if per_state[s][sample_id]["ground_truth"] == "True"),
            "unknown",
        )
        ground_truths.append(truth)
        predictions.append(prediction)
        combined_rows.append({
            "sample_id": sample_id,
            "ground_truth": truth,
            "prediction": prediction,
            "outcome": outcome,
            "votes": "|".join(votes),
            "n_votes": len(votes),
            **{f"said_{s}": per_state[s][sample_id]["prediction"] for s in WESAD_STATE_VALUES},
        })

    if "unknown" in ground_truths:
        raise SystemExit(
            "Some windows had no positive ground truth in any run — the four "
            "runs are not from the same dataset build."
        )

    metrics = WESADStateTask().aggregate_metrics(ground_truths, predictions)
    n = len(combined_rows)
    metrics["n_resolved"] = tally["resolved"]
    metrics["n_conflict"] = tally["conflict"]
    metrics["n_abstain"] = tally["abstain"]
    metrics["conflict_rate"] = round(tally["conflict"] / n, 6)
    metrics["abstain_rate"] = round(tally["abstain"] / n, 6)

    if args.out:
        out_dir = Path(args.out)
    else:
        model_name = args.model or next(
            p.name for p in sorted((results_dir / "wesad_stress").iterdir()) if p.is_dir()
        )
        out_dir = (
            results_dir / "wesad_ovr" / model_name
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(combined_rows[0].keys()))
        writer.writeheader()
        writer.writerows(combined_rows)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nCombined {n} windows -> {out_dir}")
    print(f"  resolved : {tally['resolved']:>5}  ({100*tally['resolved']/n:.1f}%)")
    print(f"  conflict : {tally['conflict']:>5}  ({100*tally['conflict']/n:.1f}%)  two or more True")
    print(f"  abstain  : {tally['abstain']:>5}  ({100*tally['abstain']/n:.1f}%)  all False")
    print()
    for key in ("accuracy", "macro_f1", "parse_rate", "n_evaluated", "n_unknown"):
        val = metrics.get(key)
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    print("\n  per-class F1:")
    for state, stats in metrics["per_class"].items():
        print(f"    {state:<11} f1={stats['f1']:.4f}  support={stats['support']}")


if __name__ == "__main__":
    main()
