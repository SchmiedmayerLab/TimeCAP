#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Unified evaluation entry point.

All evaluation runs go through this single script. Dataset, model, and task
are selected via CLI flags.

Usage examples
--------------
# Biological-sex prediction from one week of sensor data:
    python evaluation/run_eval.py \\
        --dataset mhc_metabolic_biological_sex \\
        --model gpt-4o \\
        --task metabolic_biological_sex

# Same, quick dev run:
    python evaluation/run_eval.py \\
        --dataset mhc_metabolic_biological_sex \\
        --model gpt-4o \\
        --task metabolic_biological_sex \\
        --max-samples 50 \\
        --max-users 200

# Local Llama on GPU:
    python evaluation/run_eval.py \\
        --dataset mhc_metabolic_biological_sex \\
        --model meta-llama/Llama-3.2-1B \\
        --task metabolic_biological_sex \\
        --device cuda

# Validation split:
    python evaluation/run_eval.py \\
        --dataset mhc_metabolic_biological_sex \\
        --model gpt-4o \\
        --task metabolic_biological_sex \\
        --split validation

Available datasets : mhc_metabolic_biological_sex, mhc_metabolic_bmi, …
Available tasks    : metabolic_biological_sex, metabolic_bmi, …
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
from time_series_datasets.mhc_metabolic_dataset import MHCMetabolicDataset

# ---------------------------------------------------------------------------
# Registries — add new datasets/tasks here as they are implemented
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "mhc_metabolic_biological_sex": MHCMetabolicDataset,
    "mhc_metabolic_bmi":            MHCMetabolicDataset,
    "mhc_metabolic_diabetes":       MHCMetabolicDataset,
    "mhc_metabolic_hypertension":   MHCMetabolicDataset,
    "mhc_metabolic_cardiovascular": MHCMetabolicDataset,
}

# Extra kwargs baked into each dataset name — merged with CLI kwargs at construction.
DATASET_DEFAULTS: dict[str, dict] = {
    "mhc_metabolic_biological_sex": {
        "label_column": "BiologicalSex",
        "label_values": ["Male", "Female"],
    },
    "mhc_metabolic_bmi": {
        "label_column": "BMI_categories",
        "label_values": ["Normal weight", "Underweight", "Overweight", "Obesity"],
    },
    "mhc_metabolic_diabetes": {
        "label_column": "Diabetes",
        "label_values": ["True", "False"],
    },
    "mhc_metabolic_hypertension": {
        "label_column": "Hypertension",
        "label_values": ["True", "False"],
    },
    "mhc_metabolic_cardiovascular": {
        "label_column": "cardiovascular_disease",
        "label_values": ["True", "False"],
    },
}

TASK_REGISTRY = {
    "metabolic_biological_sex":   lambda: MetabolicTask("BiologicalSex", "metabolic_biological_sex"),
    "metabolic_bmi":              lambda: MetabolicTask("BMI_categories", "metabolic_bmi"),
    "metabolic_diabetes":         lambda: MetabolicTask("Diabetes", "metabolic_diabetes"),
    "metabolic_hypertension":     lambda: MetabolicTask("Hypertension", "metabolic_hypertension"),
    "metabolic_cardiovascular":   lambda: MetabolicTask("cardiovascular_disease", "metabolic_cardiovascular"),
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
        "--max-users", type=int, default=None,
        help="Cap on users scanned when building the dataset index.",
    )
    parser.add_argument(
        "--dataset-dir", default="data/mhc_dev_subset_weekly_hf",
        help="Path to the weekly HuggingFace MHC dataset directory.",
    )
    parser.add_argument(
        "--labels-path", default="data/labels_dev_subset.parquet",
        help="Path to the MHC labels parquet file (used by metabolic datasets).",
    )
    parser.add_argument(
        "--captions-dir", default=None,
        help=(
            "Path to a directory containing captions.parquet "
            "(columns: user_id, week_start, caption). "
            "When set, captions are prepended to the prompt for each sample. "
            "Generate with evaluation/generate_captions_placeholder.py or replace "
            "with trained captionizer output. Only used for mhc_metabolic_* datasets."
        ),
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


def _build_dataset(args: argparse.Namespace):
    dataset_cls = DATASET_REGISTRY[args.dataset]
    kwargs: dict = {**DATASET_DEFAULTS.get(args.dataset, {})}
    kwargs["split"] = args.split
    kwargs["dataset_dir"] = args.dataset_dir
    if args.max_users is not None:
        kwargs["max_users"] = args.max_users
    kwargs["labels_path"] = args.labels_path
    if args.captions_dir is not None:
        kwargs["captions_dir"] = args.captions_dir
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
