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
from time_series_datasets.mimic_mortality_qa_dataset import (
    MHC_LAYOUT_LABEL_MODES,
    MIMIC_INPUT_SETS,
    MIMIC_LABEL_SPACES,
    MIMIC_POSITIVE_CLASSES,
)

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
        "--mhc-layout",
        action="store_true",
        help=(
            "Present the stay as 19 channels x 168 steps in MHC slot order, the "
            "geometry every training sample had. Heart rate keeps its true slot; "
            "the other chart channels are packed into the remaining continuous "
            "slots under MHC labels, so this varies shape while making channel "
            "semantics wrong. Mutually exclusive with --channels."
        ),
    )
    parser.add_argument(
        "--mhc-layout-labels",
        default="mhc",
        choices=list(MHC_LAYOUT_LABEL_MODES),
        help=(
            "With --mhc-layout, how the occupied slots are named. 'mhc' uses the "
            "training channel names, so shape is right and meaning is wrong; "
            "'native' names the MIMIC channel actually present. Comparing the "
            "two separates the effect of geometry from the effect of semantics."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help=(
            "Root for evaluation outputs. Runs that differ only in --channels "
            "share a task and checkpoint, so they collide under one root; point "
            "restricted-channel runs somewhere else to keep them apart."
        ),
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help=(
            "Restrict input to these chart channels, e.g. --channels heart_rate. "
            "Of the eight, only heart rate also appears in MHC training, so this "
            "is how to ask what the model does without out-of-distribution "
            "signals alongside it. Defaults to all eight."
        ),
    )
    parser.add_argument(
        "--input-set",
        default="chart",
        choices=list(MIMIC_INPUT_SETS),
        help=(
            "Which model inputs to use. 'chart' preserves the original "
            "eight-chart-channel baseline; 'chart_labs' appends a lean lab "
            "panel from labevents using the same first-window hourly bins."
        ),
    )
    parser.add_argument(
        "--run-stamp",
        default=None,
        help=(
            "Shared %%Y%%m%%d_%%H%%M%%S stamp naming this launch's output dirs. "
            "Sharded runs must pass one value to every shard so they can be "
            "recognised as one set; defaults to this process's start time."
        ),
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
    parser.add_argument(
        "--positive-class",
        default="died",
        choices=list(MIMIC_POSITIVE_CLASSES),
        help=(
            "With --label-space truefalse, which outcome True names. 'died' "
            "asks whether the patient died; 'survived' asks whether they were "
            "discharged alive. Everything else is held fixed, so running both "
            "separates question form from signal: a True-biased model scores "
            "near the majority rate under 'survived' and near zero under "
            "'died' while having learned nothing. Ignored for --label-space "
            "native."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from evaluation.evaluator import Evaluator
    from evaluation.io.writer import ResultsWriter
    from models.opentslm_native import OpenTSLMConfig, OpenTSLMModel
    from time_series_datasets.mimic_mortality_qa_dataset import MIMICMortalityQADataset

    task = MIMICMortalityTask(args.label_space, args.positive_class)
    dataset = MIMICMortalityQADataset(
        data_dir=args.mimic_data_dir,
        observation_hours=args.observation_hours,
        bin_hours=args.bin_hours,
        min_observed_channels=args.min_observed_channels,
        require_full_observation_window=args.require_full_observation_window,
        chunk_rows=args.chunk_rows,
        label_space=args.label_space,
        positive_class=args.positive_class,
        input_set=args.input_set,
        channels=args.channels,
        mhc_layout=args.mhc_layout,
        mhc_layout_labels=args.mhc_layout_labels,
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
    if args.label_space == "truefalse":
        print(f"True means            : {args.positive_class}")
    print(f"Label counts          : {summary['label_counts']}")
    print(f"Input set             : {summary['input_set']}")
    print(f"Channel count         : {summary['channel_count']}")
    print(f"Input shape           : {summary['input_shape']}")
    print()

    writer = ResultsWriter(args.results_dir)
    safe_task = re.sub(r"[^\w\-.]", "_", task.name)
    safe_model = re.sub(r"[^\w\-.]", "_", model_name)
    # The suffix separates shards that would otherwise race for one directory.
    # The stamp itself must be identical across shards, since merge_eval_shards
    # groups a launch by it: taking it here would stamp each shard with the time
    # its own dataset build finished, which varies by seconds, splitting one
    # launch across stamps so no group looks complete. The launcher passes
    # --run-stamp for that reason; the fallback only fits unsharded runs.
    stamp = args.run_stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
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
