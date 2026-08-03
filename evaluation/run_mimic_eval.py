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

from evaluation.tasks.mimic_mortality import MIMICMortalityTask

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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from evaluation.evaluator import Evaluator
    from evaluation.io.writer import ResultsWriter
    from models.opentslm_native import OpenTSLMConfig, OpenTSLMModel
    from time_series_datasets.mimic_mortality_qa_dataset import MIMICMortalityQADataset

    task = MIMICMortalityTask()
    dataset = MIMICMortalityQADataset(
        data_dir=args.mimic_data_dir,
        observation_hours=args.observation_hours,
        bin_hours=args.bin_hours,
        min_observed_channels=args.min_observed_channels,
        require_full_observation_window=args.require_full_observation_window,
        chunk_rows=args.chunk_rows,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "No MIMIC-IV mortality samples found. Check --mimic-data-dir, "
            "--observation-hours, --bin-hours, and --min-observed-channels."
        )

    config = OpenTSLMConfig(
        llm_id=args.model,
        checkpoint=args.opentslm_checkpoint,
        device=args.device,
    )
    model = OpenTSLMModel(config=config)
    model_name = f"opentslm-{Path(args.opentslm_checkpoint).stem}"

    summary = dataset.cohort_summary()
    print(f"Task                  : {task.name}")
    print(f"Model                 : {args.model}")
    print(f"Checkpoint            : {args.opentslm_checkpoint}")
    print(f"MIMIC data dir        : {args.mimic_data_dir}")
    print(f"Observation hours     : {summary['observation_hours']}")
    print(f"Bin hours             : {summary['bin_hours']}")
    print(f"Min observed channels : {summary['min_observed_channels']}")
    print(f"Require full window   : {summary['require_full_observation_window']}")
    print(f"Samples               : {summary['samples']}")
    print(f"Label counts          : {summary['label_counts']}")
    print(f"Channel count         : {summary['channel_count']}")
    print(f"Input shape           : {summary['input_shape']}")
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
