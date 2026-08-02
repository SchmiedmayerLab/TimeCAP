#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Run an eval grid across all local GPUs.

Every (checkpoint, task) pair is an independent ``run_eval.py`` process pinned
to one GPU via ``CUDA_VISIBLE_DEVICES``. Jobs are handed out from a shared
queue, so a worker that finishes a short task picks up the next one instead of
idling behind a slow neighbour.

    python scripts/gpu_sweep.py --checkpoints-dir <dir> --dry-run
    python scripts/gpu_sweep.py --checkpoints-dir <dir> --batch-size 32

Re-running skips any job that already has a complete ``metrics.json``, so an
interrupted sweep resumes by re-issuing the same command. Use ``--force`` to
re-run regardless.

Note on the ablation: the caption family lives entirely in the checkpoint. Eval
prompts are built from the channel signals and the task question and never
include captions, so ``CAPTION_TYPES`` — which selects caption families at
training time — has no effect here and is deliberately not set.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.run_eval import TASK_REGISTRY, resolve_data_paths  # noqa: E402

# Ablation checkpoints, in the order they should appear in the results table.
DEFAULT_CHECKPOINTS = [
    "joint_structural_1b_final",
    "joint_semantic_1b_final",
    "joint_crosschannel_1b_final",
    "joint_bs2_final_1b",
]


@dataclass
class Job:
    checkpoint: Path
    task: str
    llm_id: str

    @property
    def name(self) -> str:
        return f"{self.checkpoint.stem}__{self.task}"

    @property
    def model_dir_name(self) -> str:
        """Mirrors run_eval's display name, which becomes the results subdir."""
        return re.sub(r"[^\w\-.]", "_", f"opentslm-{self.checkpoint.stem}")


@dataclass
class Result:
    job: Job
    gpu: int
    returncode: int
    seconds: float
    skipped: bool = False
    metrics: dict = field(default_factory=dict)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def find_completed_run(
    results_dir: Path, task: str, model_dir_name: str, args: argparse.Namespace
) -> Path | None:
    """Newest run dir for this (task, model) that this sweep should not redo.

    A run counts only if it was produced under the same protocol. ``run_eval``
    writes a fresh timestamped dir per run and ``--save-every`` rewrites
    partials into it, so a metrics.json alone proves nothing: it may be a
    partial, or — as with the earlier capped MAX_SAMPLES ablation runs — a
    complete run under settings that aren't comparable to this sweep's.
    Runs predating run_config.json are always re-run.
    """
    model_dir = (
        results_dir
        / re.sub(r"[^\w\-.]", "_", TASK_REGISTRY[task]().name)
        / model_dir_name
    )
    if not model_dir.is_dir():
        return None

    export_dir, labels_path, split_file = resolve_data_paths()

    for run_dir in sorted(model_dir.iterdir(), reverse=True):
        metrics = _read_json(run_dir / "metrics.json")
        config = _read_json(run_dir / "run_config.json")
        if not metrics or not config:
            continue
        if config.get("max_new_tokens") != args.max_new_tokens:
            continue
        if config.get("max_samples") is not None:
            continue  # capped run; this sweep evaluates full test sets
        # Same checkpoint and task over a different split or label file is a
        # different measurement, and nothing in metrics.json says so — the May
        # runs and the canonical-split runs are indistinguishable without this.
        if (config.get("split_file"), config.get("labels_path"),
                config.get("export_dir")) != (split_file, labels_path, export_dir):
            continue
        # Partial saves land in the same dir, so require every sample.
        if metrics.get("n_evaluated") != config.get("dataset_size"):
            continue
        return run_dir
    return None


def build_command(job: Job, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "evaluation" / "run_eval.py"),
        "--task", job.task,
        "--backend", "opentslm",
        "--model", job.llm_id,
        "--opentslm-checkpoint", str(job.checkpoint),
        "--batch-size", str(args.batch_size),
        "--max-new-tokens", str(args.max_new_tokens),
        "--save-every", str(args.save_every),
    ]


def run_job(job: Job, gpu: int, args: argparse.Namespace) -> Result:
    log_path = args.log_dir / f"{job.name}.log"
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        # Eval reads the test split only; skip formatting train/validation.
        "MHC_ONLY_LOAD_SPLIT": "test",
        # Each worker is already a whole GPU's worth of work; letting torch
        # spin up a thread pool per process just oversubscribes the host.
        "OMP_NUM_THREADS": str(args.threads_per_worker),
    }
    env.pop("CAPTION_TYPES", None)  # see module docstring
    # This sweep always evaluates the full test set. MAX_SAMPLES is read
    # straight from the environment by run_eval, and the earlier ablation
    # runner exported it, so an inherited value would silently truncate every
    # job — and, because run_config records the cap, make every job look
    # incomparable on the next resume and re-run the whole grid.
    env.pop("MAX_SAMPLES", None)

    started = time.monotonic()
    with open(log_path, "w") as log:
        proc = subprocess.run(
            build_command(job, args), env=env, stdout=log, stderr=subprocess.STDOUT
        )
    elapsed = time.monotonic() - started

    run_dir = find_completed_run(args.results_dir, job.task, job.model_dir_name, args)
    metrics = _read_json(run_dir / "metrics.json") if run_dir else None
    return Result(job=job, gpu=gpu, returncode=proc.returncode, seconds=elapsed,
                  metrics=metrics or {})


def worker(gpu: int, jobs: queue.Queue, results: list, lock: threading.Lock,
           args: argparse.Namespace) -> None:
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        with lock:
            print(f"[gpu {gpu}] START {job.name}", flush=True)
        result = run_job(job, gpu, args)
        with lock:
            status = "DONE " if result.returncode == 0 else "FAIL "
            # n_evaluated, not n_samples: the latter counts only samples whose
            # output parsed, which is 0 on the tasks where a checkpoint answers
            # off-label — indistinguishable from a job that produced nothing.
            print(f"[gpu {gpu}] {status}{job.name}  "
                  f"rc={result.returncode}  {result.seconds / 60:.1f} min  "
                  f"n={result.metrics.get('n_evaluated', '?')} "
                  f"parsed={result.metrics.get('parse_rate', '?')}", flush=True)
            results.append(result)
        jobs.task_done()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an eval grid across all local GPUs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints-dir", type=Path, required=True,
        help="Directory holding the .pt checkpoints named by --checkpoints.",
    )
    parser.add_argument(
        "--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS,
        help="Checkpoint stems (without .pt) to evaluate.",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=sorted(TASK_REGISTRY),
        help="Tasks to evaluate.",
    )
    parser.add_argument("--llm-id", default="meta-llama/Llama-3.2-1B")
    parser.add_argument(
        "--gpus", type=int, default=None,
        help="Number of GPUs. Defaults to torch.cuda.device_count().",
    )
    parser.add_argument(
        "--workers-per-gpu", type=int, default=1,
        help="Concurrent jobs per GPU. A 1B model leaves headroom on an 80GB "
             "card, but batching inside a job beats running more of them.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-new-tokens", type=int, default=32,
        help="Greedy decoding makes a cap equivalent to truncating the full "
             "generation, so its cost was measured directly against the "
             "existing predictions: for the four 1B caption-family checkpoints "
             "a cap of 32 leaves parse rate unchanged on all 12 tasks. Raise to "
             "64 if the 3B checkpoints are added to the grid.",
    )
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Where run_eval writes results, used for the resume check. "
             "run_eval always writes to ./results relative to its working "
             "directory, so override this only if you also change that.",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("sweep_logs"))
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run jobs that already have complete results.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the jobs that would run, then exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.gpus is None:
        import torch
        args.gpus = torch.cuda.device_count()
    if args.gpus < 1:
        print("No GPUs visible — set --gpus explicitly to run anyway.", file=sys.stderr)
        return 1

    unknown = sorted(set(args.tasks) - set(TASK_REGISTRY))
    if unknown:
        print(f"Unknown tasks: {unknown}\nAvailable: {sorted(TASK_REGISTRY)}",
              file=sys.stderr)
        return 1

    missing = [
        stem for stem in args.checkpoints
        if not (args.checkpoints_dir / f"{stem}.pt").is_file()
    ]
    if missing:
        print(f"Missing checkpoints in {args.checkpoints_dir}: {missing}",
              file=sys.stderr)
        return 1

    pending: list[Job] = []
    skipped: list[Job] = []
    for stem in args.checkpoints:
        for task in args.tasks:
            job = Job(args.checkpoints_dir / f"{stem}.pt", task, args.llm_id)
            existing = None if args.force else find_completed_run(
                args.results_dir, task, job.model_dir_name, args
            )
            (skipped if existing else pending).append(job)

    total = len(pending) + len(skipped)
    print(f"Grid: {len(args.checkpoints)} checkpoints x {len(args.tasks)} tasks "
          f"= {total} jobs")
    print(f"  {len(skipped)} already complete, {len(pending)} to run")
    print(f"  {args.gpus} GPUs x {args.workers_per_gpu} workers, "
          f"batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens}")

    if args.dry_run:
        for job in pending:
            print(f"  RUN  {job.name}")
        for job in skipped:
            print(f"  SKIP {job.name}")
        return 0

    if not pending:
        print("Nothing to do.")
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)

    jobs: queue.Queue = queue.Queue()
    # Interleave by task so each checkpoint's slow tasks spread across workers
    # instead of piling onto one at the end.
    for job in sorted(pending, key=lambda j: (j.task, j.checkpoint.stem)):
        jobs.put(job)

    results: list[Result] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(gpu, jobs, results, lock, args), daemon=True)
        for gpu in range(args.gpus)
        for _ in range(args.workers_per_gpu)
    ]

    started = time.monotonic()
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\nInterrupted — finished jobs keep their results; "
              "re-run the same command to resume.", file=sys.stderr)
        return 130
    elapsed = time.monotonic() - started

    failures = [r for r in results if r.returncode != 0]
    manifest = args.log_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "elapsed_minutes": round(elapsed / 60, 2),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "jobs": [
            {
                "name": r.job.name,
                "task": r.job.task,
                "checkpoint": r.job.checkpoint.stem,
                "gpu": r.gpu,
                "returncode": r.returncode,
                "minutes": round(r.seconds / 60, 2),
                "metrics": r.metrics,
            }
            for r in sorted(results, key=lambda r: r.job.name)
        ],
    }, indent=2))

    print(f"\nSweep finished in {elapsed / 60:.1f} min — "
          f"{len(results) - len(failures)}/{len(results)} succeeded")
    print(f"Manifest: {manifest}")
    for failure in failures:
        print(f"  FAILED {failure.job.name} (rc={failure.returncode}) — "
              f"see {args.log_dir / (failure.job.name + '.log')}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
