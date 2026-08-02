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

from evaluation.tasks.wesad_state import WESADStateTask

DEFAULT_WESAD_TIMEF_DATASET_DIR = "data/wesad_timef"


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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from evaluation.evaluator import Evaluator
    from evaluation.io.writer import ResultsWriter
    from models.opentslm_native import OpenTSLMConfig, OpenTSLMModel
    from time_series_datasets.wesad_state_qa_dataset import WESADStateQADataset

    task = WESADStateTask()
    dataset = WESADStateQADataset(
        dataset_dir=args.wesad_dataset_dir,
        window_seconds=args.window_seconds,
        min_label_purity=args.min_label_purity,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "No WESAD windows found. Check --wesad-dataset-dir, "
            "--window-seconds, and --min-label-purity."
        )

    config = OpenTSLMConfig(
        llm_id=args.model,
        checkpoint=args.opentslm_checkpoint,
        device=args.device,
    )
    model = OpenTSLMModel(config=config)
    model_name = f"opentslm-{Path(args.opentslm_checkpoint).stem}"

    print(f"Task             : {task.name}")
    print(f"Model            : {args.model}")
    print(f"Checkpoint       : {args.opentslm_checkpoint}")
    print(f"WESAD dataset    : {args.wesad_dataset_dir}")
    print(f"Window seconds   : {args.window_seconds}")
    print(f"Min label purity : {args.min_label_purity}")
    print(f"Samples          : {len(dataset)}")
    print()

    writer = ResultsWriter()
    safe_task = re.sub(r"[^\w\-.]", "_", task.name)
    safe_model = re.sub(r"[^\w\-.]", "_", model_name)
    out_dir = (
        writer.results_dir
        / safe_task
        / safe_model
        / datetime.now().strftime("%Y%m%d_%H%M%S")
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
