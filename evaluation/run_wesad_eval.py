#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Evaluate MHC-trained OpenTSLM weights on WESAD state labels."""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.tasks.wesad_state import WESADBinaryTask, WESADStateTask
from time_series_datasets.wesad_state_qa_dataset import WESAD_BINARY_QUESTIONS

DEFAULT_WESAD_TIMEF_DATASET_DIR = "data/wesad_timef"


class _ShardView:
    """Every ``num_shards``-th window, offset by ``shard_index``.

    Strided rather than contiguous on purpose. Windows come out in WESAD
    protocol order — a long baseline block, then stress, then meditation, then
    amusement — so contiguous blocks would hand one shard nothing but baseline
    and another nothing but stress. Each shard's partial metrics would then be
    computed over a near-single-class subset and be meaningless on its own;
    striding keeps every shard roughly class-balanced. Merge the per-shard
    predictions.csv files to get the dataset-level numbers either way.
    """

    def __init__(self, dataset, num_shards: int, shard_index: int) -> None:
        if num_shards < 1:
            raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
        if not 0 <= shard_index < num_shards:
            raise ValueError(
                f"--shard-index must be in [0, {num_shards}), got {shard_index}"
            )
        self._dataset = dataset
        self._idxs = list(range(shard_index, len(dataset), num_shards))

    def __len__(self) -> int:
        return len(self._idxs)

    def __getitem__(self, idx: int):
        return self._dataset[self._idxs[idx]]

    def get_eval_context(self, idx: int):
        return self._dataset.get_eval_context(self._idxs[idx])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 4-way WESAD state evaluation with an MHC-trained OpenTSLM "
            "checkpoint."
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
        "--wesad-dataset-dir",
        default=os.environ.get(
            "WESAD_TIMEF_DATASET_DIR",
            DEFAULT_WESAD_TIMEF_DATASET_DIR,
        ),
        help="Path to the stored WESAD timef/HuggingFace dataset.",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=120,
        help="Non-overlapping WESAD sensor window length.",
    )
    parser.add_argument(
        "--min-label-purity",
        type=float,
        default=0.8,
        help="Minimum majority-label fraction required to keep a window.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap evaluated samples for smoke tests.",
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
        help="Split the windows across this many processes (one per GPU).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard this process evaluates, in [0, --num-shards).",
    )
    parser.add_argument(
        "--binary-label",
        default=None,
        choices=sorted(WESAD_BINARY_QUESTIONS),
        help=(
            "Ask a yes/no question about this one state instead of the 4-way "
            "question. An MHC-trained checkpoint has never seen the 4-way label "
            "space and answers True/False regardless of the options offered, so "
            "the 4-way parse rate collapses to ~0; this framing is scorable. "
            "Run once per state and combine with scripts/combine_wesad_ovr.py."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from evaluation.evaluator import Evaluator
    from evaluation.io.writer import ResultsWriter
    from models.opentslm_native import OpenTSLMConfig, OpenTSLMModel
    from time_series_datasets.wesad_state_qa_dataset import WESADStateQADataset

    task = (
        WESADBinaryTask(args.binary_label)
        if args.binary_label
        else WESADStateTask()
    )
    dataset = WESADStateQADataset(
        dataset_dir=args.wesad_dataset_dir,
        window_seconds=args.window_seconds,
        min_label_purity=args.min_label_purity,
        binary_label=args.binary_label,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "No WESAD windows found. Check --wesad-dataset-dir, "
            "--window-seconds, and --min-label-purity."
        )

    n_total = len(dataset)
    if args.num_shards > 1:
        dataset = _ShardView(dataset, args.num_shards, args.shard_index)
        if len(dataset) == 0:
            raise RuntimeError(
                f"Shard {args.shard_index}/{args.num_shards} is empty "
                f"({n_total} windows total) — use fewer shards."
            )

    config = OpenTSLMConfig(
        llm_id=args.model,
        checkpoint=args.opentslm_checkpoint,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    model = OpenTSLMModel(config=config)
    model_name = f"opentslm-{Path(args.opentslm_checkpoint).stem}"

    print(f"Task             : {task.name}")
    print(f"Model            : {args.model}")
    print(f"Checkpoint       : {args.opentslm_checkpoint}")
    print(f"WESAD dataset    : {args.wesad_dataset_dir}")
    print(f"Window seconds   : {args.window_seconds}")
    print(f"Min label purity : {args.min_label_purity}")
    if args.num_shards > 1:
        print(f"Shard            : {args.shard_index}/{args.num_shards}")
        print(f"Samples          : {len(dataset)} of {n_total}")
    else:
        print(f"Samples          : {len(dataset)}")
    print()

    writer = ResultsWriter()
    safe_task = re.sub(r"[^\w\-.]", "_", task.name)
    safe_model = re.sub(r"[^\w\-.]", "_", model_name)
    # Shards are separate processes started at the same second, so the timestamp
    # alone does not separate them — without the suffix they would race for one
    # directory and overwrite each other's predictions.
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
