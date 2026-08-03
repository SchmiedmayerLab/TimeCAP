#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Evaluate MHC-trained OpenTSLM weights on MIMIC-IV mortality."""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.sharding import ShardView
from evaluation.tasks.mimic_mortality import MIMICMortalityTask
from time_series_datasets.mimic_mortality_qa_dataset import MIMIC_LABEL_SPACES

DEFAULT_MIMIC_DATA_DIR = "data/mimic-iv-clinical-database-demo-2.2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MIMIC-IV in-hospital mortality evaluation with an "
            "MHC-trained OpenTSLM checkpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Backbone LLM id used with the local OpenTSLM checkpoint.",
    )
    parser.add_argument(
        "--opentslm-checkpoint",
        required=True,
        help="Path to locally trained OpenTSLM-SP weights.",
    )
    parser.add_argument(
        "--mimic-data-dir",
        default=os.environ.get("MIMIC_DATA_DIR", DEFAULT_MIMIC_DATA_DIR),
        help="MIMIC-IV root containing flat demo files or hosp/ and icu/ folders.",
    )
    parser.add_argument(
        "--observation-hours",
        type=int,
        default=24,
        help="Hours after ICU admission used as model input.",
    )
    parser.add_argument(
        "--bin-hours",
        type=int,
        default=1,
        help="Width of each chart-event aggregation bin in hours.",
    )
    parser.add_argument(
        "--min-observed-channels",
        type=int,
        default=1,
        help="Minimum selected chart channels observed in the input window.",
    )
    parser.add_argument(
        "--require-full-observation-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require ICU outtime to be at least intime + observation-hours. "
            "This reduces censoring leakage from early stopped charting."
        ),
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=1_000_000,
        help="Rows per chartevents chunk while building the dataset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap evaluated samples for smoke tests after dataset construction.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Persist partial results every N samples.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override, e.g. cuda, mps, or cpu.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help=(
            "Generation cap. The answer is a single label, so a low value costs "
            "nothing and dominates runtime on CPU — but leave headroom for "
            "checkpoints that preface the answer with a caption."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Samples per generate call. Leave at 1 on CPU.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the stays across this many processes (one per GPU).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard this process evaluates, in [0, --num-shards).",
    )
    parser.add_argument(
        "--label-space",
        default="native",
        choices=list(MIMIC_LABEL_SPACES),
        help=(
            "Answer vocabulary. 'native' asks for survived/died; 'truefalse' "
            "asks the same question with True/False, a space an MHC-trained "
            "checkpoint has seen. Unseen label spaces collapse to parse rate ~0 "
            "regardless of the options listed, so 'native' measures that gap "
            "and 'truefalse' is the scorable framing."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from evaluation.evaluator import Evaluator
    from evaluation.io.writer import ResultsWriter
    from models.opentslm_native import OpenTSLMConfig, OpenTSLMModel
    from time_series_datasets.mimic_mortality_qa_dataset import MIMICMortalityQADataset

    task = MIMICMortalityTask(args.label_space)
    dataset = MIMICMortalityQADataset(
        data_dir=args.mimic_data_dir,
        observation_hours=args.observation_hours,
        bin_hours=args.bin_hours,
        min_observed_channels=args.min_observed_channels,
        require_full_observation_window=args.require_full_observation_window,
        chunk_rows=args.chunk_rows,
        label_space=args.label_space,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "No MIMIC-IV mortality samples found. Check --mimic-data-dir, "
            "--observation-hours, --bin-hours, and --min-observed-channels."
        )

    n_total = len(dataset)
    summary = dataset.cohort_summary()
    if args.num_shards > 1:
        dataset = ShardView(dataset, args.num_shards, args.shard_index)
        if len(dataset) == 0:
            raise RuntimeError(
                f"Shard {args.shard_index}/{args.num_shards} is empty "
                f"({n_total} stays total) — use fewer shards."
            )

    config = OpenTSLMConfig(
        llm_id=args.model,
        checkpoint=args.opentslm_checkpoint,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    model = OpenTSLMModel(config=config)
    model_name = f"opentslm-{Path(args.opentslm_checkpoint).stem}"

    print(f"Task                  : {task.name}")
    print(f"Model                 : {args.model}")
    print(f"Checkpoint            : {args.opentslm_checkpoint}")
    print(f"MIMIC data dir        : {args.mimic_data_dir}")
    print(f"Observation hours     : {summary['observation_hours']}")
    print(f"Bin hours             : {summary['bin_hours']}")
    print(f"Min observed channels : {summary['min_observed_channels']}")
    print(f"Require full window   : {summary['require_full_observation_window']}")
    if args.num_shards > 1:
        print(f"Shard                 : {args.shard_index}/{args.num_shards}")
        print(f"Samples               : {len(dataset)} of {n_total}")
    else:
        print(f"Samples               : {summary['samples']}")
    print(f"Label space           : {args.label_space}")
    print(f"Label counts          : {summary['label_counts']}")
    print(f"Channel count         : {summary['channel_count']}")
    print(f"Input shape           : {summary['input_shape']}")
    print()

    writer = ResultsWriter()
    safe_task = re.sub(r"[^\w\-.]", "_", task.name)
    safe_model = re.sub(r"[^\w\-.]", "_", model_name)
    # Shards start in the same second, so the timestamp alone does not separate
    # them — without the suffix they would race for one directory.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.num_shards > 1:
        stamp = f"{stamp}_shard{args.shard_index:02d}of{args.num_shards:02d}"
    out_dir = (
        writer.results_dir
        / safe_task
        / safe_model
        / stamp
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    result = Evaluator().run(
        task=task,
        model=model,
        dataset=dataset,
        model_name=model_name,
        max_samples=args.max_samples,
        save_every=args.save_every,
        results_writer=writer,
        out_dir=out_dir,
        batch_size=args.batch_size,
    )
    writer.write(result, out_dir=out_dir)

    print("\n=== RESULTS ===")
    for key, val in result.metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
