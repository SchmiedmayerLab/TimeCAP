#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Unified evaluation entry point.

Three backends, selected via ``--backend``:

* ``openai`` (default) — VLM over OpenAI-compatible API (GPT-4o, Ollama, vLLM, …).
* ``vlm-local``        — VLM via HF ``image-text-to-text`` pipeline (Gemma-VL, Qwen-VL, …).
* ``opentslm``         — Native OpenTSLM-SP (requires ``--opentslm-checkpoint``).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluator import Evaluator
from evaluation.io.writer import ResultsWriter
from evaluation.tasks.metabolic import MetabolicTask
from mhc_weekly.constants import WEEKLY_CHANNEL_CONFIG
from models.client import ClientConfig, ClientModel
from models.local import LocalConfig, LocalModel
from models.opentslm_native import OpenTSLMConfig, OpenTSLMModel

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "biological_sex":  lambda: MetabolicTask("BiologicalSex"),
    "bmi":             lambda: MetabolicTask("BMI_categories"),
    "diabetes":        lambda: MetabolicTask("Diabetes"),
    "hypertension":    lambda: MetabolicTask("Hypertension"),
    "cardiovascular":  lambda: MetabolicTask("cardiovascular_disease"),
    "work":            lambda: MetabolicTask("work"),
    "sleep_diagnosis": lambda: MetabolicTask("sleep_diagnosis1"),
    "happiness":       lambda: MetabolicTask("happiness_categories"),
}

BACKENDS = ("openai", "vlm-local", "opentslm")

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
        help='Model identifier (meaning depends on --backend).',
    )
    parser.add_argument(
        "--backend", default="openai", choices=BACKENDS,
        help="Model backend.",
    )
    parser.add_argument(
        "--opentslm-checkpoint", default=None,
        help=(
            "Path to locally trained OpenTSLM-SP weights (opentslm backend). "
            "When set, --model is the backbone LLM id. When omitted, --model "
            "is treated as a HF Hub repo_id and pulled via "
            "OpenTSLM.load_pretrained."
        ),
    )
    return parser.parse_args()


def _build_model(args: argparse.Namespace):
    if args.backend == "openai":
        config = ClientConfig(model=args.model)  # base_url via OPENAI_BASE_URL env
        return ClientModel(config=config, channel_config=WEEKLY_CHANNEL_CONFIG), args.model
    if args.backend == "vlm-local":
        config = LocalConfig(model=args.model)
        return LocalModel(config=config, channel_config=WEEKLY_CHANNEL_CONFIG), args.model.split("/")[-1]
    if args.backend == "opentslm":
        if args.opentslm_checkpoint:
            config = OpenTSLMConfig(
                llm_id=args.model, checkpoint=args.opentslm_checkpoint
            )
            display = f"opentslm-{Path(args.opentslm_checkpoint).stem}"
        else:
            config = OpenTSLMConfig(repo_id=args.model)
            display = args.model.split("/")[-1]
        return OpenTSLMModel(config=config), display
    raise ValueError(f"Unknown backend: {args.backend}")


def _build_dataset(args: argparse.Namespace, task: MetabolicTask):
    return task.dataset_cls(
        split="test",
        EOS_TOKEN="",
        target_label=task._label_column,
        export_dir=os.environ.get("MHC_EXPORT_DIR", "exports"),
        label_lookup_path=os.environ.get(
            "MHC_LABELS_PATH", "data/labels_dev_subset.parquet"
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    max_samples = int(os.environ["MAX_SAMPLES"]) if os.environ.get("MAX_SAMPLES") else None

    print(f"Task     : {args.task}")
    print(f"Backend  : {args.backend}")
    print(f"Model    : {args.model}")
    print()

    task = TASK_REGISTRY[args.task]()
    model, model_name = _build_model(args)
    dataset = _build_dataset(args, task)
    if len(dataset) == 0:
        print("No samples found — check MHC_EXPORT_DIR.")
        sys.exit(1)

    evaluator = Evaluator()
    result = evaluator.run(
        task=task,
        model=model,
        dataset=dataset,
        model_name=model_name,
        max_samples=max_samples,
    )

    ResultsWriter().write(result)

    print("\n=== RESULTS ===")
    for key, val in result.metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
