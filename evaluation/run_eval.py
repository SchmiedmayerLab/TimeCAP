#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Unified evaluation entry point.

All evaluation runs go through this single script. Dataset, model, and task
are selected via CLI flags — the script itself contains no task-specific logic.

Usage examples
--------------
# Activity recognition with GPT-4o (full test split):
    python evaluation/run_eval.py \\
        --dataset mhc_activity \\
        --model gpt-4o \\
        --task activity

# Same, quick dev run:
    python evaluation/run_eval.py \\
        --dataset mhc_activity \\
        --model gpt-4o \\
        --task activity \\
        --max-samples 50 \\
        --max-recordings 500

# Local Llama on GPU:
    python evaluation/run_eval.py \\
        --dataset mhc_activity \\
        --model meta-llama/Llama-3.2-1B \\
        --task activity \\
        --device cuda

# Validation split:
    python evaluation/run_eval.py \\
        --dataset mhc_activity \\
        --model gpt-4o \\
        --task activity \\
        --split validation

Available datasets : mhc_activity
Available tasks    : activity
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluator import Evaluator
from evaluation.io.writer import ResultsWriter
from evaluation.tasks.activity import ActivityTask
from mhc.constants import MHC_CHANNEL_CONFIG
from models.client import ClientConfig, ClientModel
from models.local import LocalConfig, LocalModel
from time_series_datasets.mhc_activity_dataset import MHCActivityDataset

# ---------------------------------------------------------------------------
# Registries — add new datasets/tasks here as they are implemented
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "mhc_activity": MHCActivityDataset,
}

TASK_REGISTRY = {
    "activity": ActivityTask,
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
        "--dataset", required=True, choices=sorted(DATASET_REGISTRY),
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--task", required=True, choices=sorted(TASK_REGISTRY),
        help="Evaluation task (determines prompt and metrics).",
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
        "--max-samples", type=int, default=None,
        help="Maximum number of samples to evaluate (None = all).",
    )
    parser.add_argument(
        "--max-recordings", type=int, default=None,
        help="Cap on MHC recordings scanned when building the dataset index.",
    )
    parser.add_argument(
        "--window-size", type=int, default=60,
        help="Window duration in minutes (activity/sleep datasets).",
    )
    parser.add_argument(
        "--dataset-dir", default="data/mhc_dev_subset_hf",
        help="Path to the HuggingFace MHC dataset directory.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for local models: cuda, mps, or cpu.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Root directory for result files.",
    )
    return parser.parse_args()


def _build_model(args: argparse.Namespace):
    is_local = any(kw in args.model.lower() for kw in _LOCAL_KEYWORDS)
    if is_local:
        config = LocalConfig(model=args.model, device=args.device)
        model = LocalModel(config=config, channel_config=MHC_CHANNEL_CONFIG)
        display_name = args.model.split("/")[-1]
    else:
        config = ClientConfig(base_url=args.base_url, model=args.model)
        model = ClientModel(config=config, channel_config=MHC_CHANNEL_CONFIG)
        display_name = args.model
    return model, display_name


def _build_dataset(args: argparse.Namespace):
    dataset_cls = DATASET_REGISTRY[args.dataset]
    kwargs = dict(split=args.split, dataset_dir=args.dataset_dir)
    # Pass optional params only if the dataset class accepts them
    import inspect
    sig = inspect.signature(dataset_cls.__init__)
    if "window_size" in sig.parameters:
        kwargs["window_size"] = args.window_size
    if "max_recordings" in sig.parameters and args.max_recordings is not None:
        kwargs["max_recordings"] = args.max_recordings
    return dataset_cls(**kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print(f"Dataset  : {args.dataset} ({args.split})")
    print(f"Task     : {args.task}")
    print(f"Model    : {args.model}")
    print()

    dataset = _build_dataset(args)
    if len(dataset) == 0:
        print("No samples found — check --dataset-dir and --split.")
        sys.exit(1)

    model, model_name = _build_model(args)
    task = TASK_REGISTRY[args.task]()
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
