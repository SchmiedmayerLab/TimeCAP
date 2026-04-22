#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Unified evaluation entry point.

Usage examples::

    python evaluation/run_eval.py --dataset mhc_metabolic --task diabetes --model gpt-4o
    python evaluation/run_eval.py --dataset mhc_metabolic --task bmi --model gpt-4o --max-samples 50
    python evaluation/run_eval.py --dataset mhc_metabolic --task biological_sex --model meta-llama/Llama-3.2-1B --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluator import Evaluator
from evaluation.io.writer import ResultsWriter
from evaluation.tasks.metabolic import MetabolicTask
from mhc_weekly.constants import WEEKLY_CHANNEL_CONFIG
from models.client import ClientConfig, ClientModel
from models.local import LocalConfig, LocalModel

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "biological_sex": lambda: MetabolicTask("BiologicalSex"),
    "bmi":            lambda: MetabolicTask("BMI_categories"),
    "diabetes":       lambda: MetabolicTask("Diabetes"),
    "hypertension":   lambda: MetabolicTask("Hypertension"),
    "cardiovascular": lambda: MetabolicTask("cardiovascular_disease"),
}

_LOCAL_KEYWORDS = {"llama", "gemma", "mistral", "phi", "qwen", "falcon"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation for a dataset/model/task combination.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task", required=True, choices=sorted(TASK_REGISTRY),
        help="Evaluation task (determines dataset, prompt, and metrics).",
    )
    parser.add_argument(
        "--model", required=True,
        help='Model name, e.g. "gpt-4o", "meta-llama/Llama-3.2-1B", "llama3.2".',
    )
    parser.add_argument(
        "--base-url", default="https://api.openai.com/v1",
        help="API base URL for client models (OpenAI, Ollama, vLLM, …).",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "validation", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, metavar="N",
        help="Maximum number of samples to evaluate (positive int, None = all).",
    )
    parser.add_argument(
        "--export-dir", default="exports",
        help="Path to the Arrow shard directory (recordings_*.arrow).",
    )
    parser.add_argument(
        "--labels-path", default="data/labels_dev_subset.parquet",
        help="Path to the MHC labels parquet file (used by metabolic datasets).",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for local models: cuda, mps, or cpu.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Root directory for result files.",
    )
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be a positive integer")
    return args


def _build_model(args: argparse.Namespace):
    channel_config = WEEKLY_CHANNEL_CONFIG
    is_local = any(kw in args.model.lower() for kw in _LOCAL_KEYWORDS)
    if is_local:
        config = LocalConfig(model=args.model, device=args.device)
        model = LocalModel(config=config, channel_config=channel_config)
        display_name = args.model.split("/")[-1]  # strip org prefix for result paths only; full name is passed to model
    else:
        config = ClientConfig(base_url=args.base_url, model=args.model)
        model = ClientModel(config=config, channel_config=channel_config)
        display_name = args.model
    return model, display_name


def _build_dataset(args: argparse.Namespace, task: MetabolicTask):
    return task.dataset_cls(
        split=args.split,
        EOS_TOKEN="",
        target_label=task._label_column,
        export_dir=args.export_dir,
        label_lookup_path=args.labels_path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print(f"Task     : {args.task} ({args.split})")
    print(f"Model    : {args.model}")
    print()

    task = TASK_REGISTRY[args.task]()
    dataset = _build_dataset(args, task)
    if len(dataset) == 0:
        print("No samples found — check --export-dir and --split.")
        sys.exit(1)

    model, model_name = _build_model(args)
    evaluator = Evaluator()

    result = evaluator.run(
        task=task,
        model=model,
        dataset=dataset,
        model_name=model_name,
        max_samples=args.max_samples,
    )

    ResultsWriter(results_dir=args.results_dir).write(result)

    print("\n=== RESULTS ===")
    for key, val in result.metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
