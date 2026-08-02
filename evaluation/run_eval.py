#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
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
import json
import os
import subprocess
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
    # NEW unseen tasks
    "cad":             lambda: MetabolicTask("CAD"),
    "afib":            lambda: MetabolicTask("Atrial fibrillation (Afib)"),
    "sleep_time":      lambda: MetabolicTask("sleep_time_categories"),
    "blood_pressure":  lambda: MetabolicTask("blood_pressure_categories"),
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
    parser.add_argument(
        "--save-every", type=int, default=None,
        help=(
            "Persist partial results every N samples (and on KeyboardInterrupt) "
            "to the same output dir. Useful for long evals on slow filesystems."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help=(
            "Samples per generate call. Only the opentslm backend batches "
            "natively; other backends loop internally, so this is a no-op there."
        ),
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help=(
            "Generation cap (opentslm backend). Answers are a few tokens on 10 "
            "of the 12 tasks; only blood_pressure and sleep_time run long, and "
            "only for checkpoints that fall back to captioning on them. "
            "Measured against the existing predictions, a cap of 32 leaves "
            "parse rate identical everywhere except joint_3b_final, which loses "
            "~1.7pt on blood_pressure and ~0.3pt on sleep_time (recovered at 64)."
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
                llm_id=args.model,
                checkpoint=args.opentslm_checkpoint,
                max_new_tokens=args.max_new_tokens,
            )
            display = f"opentslm-{Path(args.opentslm_checkpoint).stem}"
        else:
            config = OpenTSLMConfig(
                repo_id=args.model, max_new_tokens=args.max_new_tokens
            )
            display = args.model.split("/")[-1]
        return OpenTSLMModel(config=config), display
    raise ValueError(f"Unknown backend: {args.backend}")


def resolve_data_paths() -> tuple[str, str, str]:
    """Resolve (export_dir, labels_path, split_file) from the environment.

    One resolution point on purpose: ``run_config.json`` records these, and a
    record that can drift from what was actually loaded is worse than no record.
    """
    split_file = os.environ.get("MHC_SPLIT_FILE")
    if not split_file:
        raise RuntimeError(
            "MHC_SPLIT_FILE env var is required: point at MHC-benchmark's "
            "sharable_users_seed42_2026.json so split assignment is consistent."
        )
    return (
        os.environ.get("MHC_EXPORT_DIR", "exports"),
        os.environ.get("MHC_LABELS_PATH", "data/labels_dev_subset.parquet"),
        split_file,
    )


def _git_commit() -> str | None:
    """Short SHA of the working tree, with a ``-dirty`` suffix if uncommitted."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if sha.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        dirty = "-dirty" if status.stdout.strip() else ""
        return sha.stdout.strip() + dirty
    except (OSError, subprocess.SubprocessError):
        return None


def _build_dataset(task: MetabolicTask):
    export_dir, labels_path, split_file = resolve_data_paths()
    return task.dataset_cls(
        split="test",
        EOS_TOKEN="",
        target_label=task._label_column,
        export_dir=export_dir,
        label_lookup_path=labels_path,
        split_file=split_file,
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
    print(f"Batch    : {args.batch_size}")
    print(f"Max tok  : {args.max_new_tokens}")
    print()

    export_dir, labels_path, split_file = resolve_data_paths()
    task = TASK_REGISTRY[args.task]()
    model, model_name = _build_model(args)
    dataset = _build_dataset(task)
    if len(dataset) == 0:
        print("No samples found — check MHC_EXPORT_DIR.")
        sys.exit(1)

    # Pre-create the output dir so periodic saves rewrite the same path.
    from datetime import datetime
    import re
    writer = ResultsWriter()
    safe_task = re.sub(r"[^\w\-.]", "_", task.name)
    safe_model = re.sub(r"[^\w\-.]", "_", model_name)
    out_dir = writer.results_dir / safe_task / safe_model / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Record how this run was produced, alongside (not inside) metrics.json so
    # existing readers of that file are unaffected. Sweeps use it to tell a
    # completed run apart from one made under different settings — a capped
    # MAX_SAMPLES run and a full-test-set run are not interchangeable, and
    # without this the only signal is "a metrics.json exists".
    #
    # The three data inputs are recorded together because they, not the model
    # arguments, are what decides whether two runs belong in the same table:
    # the same checkpoint on the same task scored a different test set before
    # and after the switch to MHC-benchmark's canonical split, and nothing in
    # the metrics distinguishes them.
    (out_dir / "run_config.json").write_text(json.dumps({
        "task": args.task,
        "label": task.name,
        "backend": args.backend,
        "model": args.model,
        "checkpoint": args.opentslm_checkpoint,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_samples": max_samples,
        "dataset_size": len(dataset),
        "split": "test",
        "export_dir": export_dir,
        "split_file": split_file,
        "labels_path": labels_path,
        "git_commit": _git_commit(),
        "started": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))

    evaluator = Evaluator()
    result = evaluator.run(
        task=task,
        model=model,
        dataset=dataset,
        model_name=model_name,
        max_samples=max_samples,
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
