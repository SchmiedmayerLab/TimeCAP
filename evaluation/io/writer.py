#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""ResultsWriter — persists EvalResult to disk.

Output layout::

    results/
        <task_name>/
            <model_name>/
                <timestamp>/
                    metrics.json      ← aggregated metrics (F1, SE, …)
                    predictions.csv   ← per-sample ground truth + prediction
"""
from __future__ import annotations

import csv
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from evaluation.evaluator import EvalResult


@contextmanager
def _atomic_write(path: Path, **open_kwargs):
    """Write *path* via a sibling temp file, swapped in with ``os.replace``.

    ``Evaluator.run(save_every=...)`` rewrites both output files in place, in
    full, while the eval is still running. Truncating the real file to write it
    means a process killed mid-write (preemption, OOM) leaves a half-written
    predictions.csv where a complete one used to be — losing exactly the work
    the periodic save exists to protect. ``os.replace`` is atomic within a
    filesystem, so a reader sees either the previous version or the new one.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp, "w", **open_kwargs) as f:
            yield f
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class ResultsWriter:
    """Writes EvalResult files under results/<task>/<model>/<timestamp>/.

    Args:
        results_dir: Root directory for all evaluation outputs.
    """

    def __init__(self, results_dir: Path | str = Path("results")) -> None:
        self.results_dir = Path(results_dir)

    def write(self, result: EvalResult, out_dir: Path | None = None) -> Path:
        """Persist *result* to disk and return the output directory path.

        If ``out_dir`` is given, write into it (replacing any existing
        metrics.json / predictions.csv atomically). Otherwise create a fresh
        ``<timestamp>`` subdir. Pre-computing ``out_dir`` once and reusing
        it lets callers do periodic partial saves without proliferating
        directories — see ``Evaluator.run(save_every=...)``.

        Creates:
            metrics.json     — aggregated metrics dict (directly JSON-serialisable).
            predictions.csv  — one row per evaluated sample.
        """
        if out_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_task = re.sub(r"[^\w\-.]", "_", result.task_name)
            safe_model = re.sub(r"[^\w\-.]", "_", result.model_name)
            out_dir = self.results_dir / safe_task / safe_model / timestamp
            out_dir.mkdir(parents=True, exist_ok=False)
        else:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        # predictions.csv
        base_fields = [
            "sample_id",
            "ground_truth",
            "prediction",
            "raw_prediction",
            "input_tokens",
            "output_tokens",
        ]
        metadata_fields: list[str] = []
        seen_metadata: set[str] = set()
        for sample in result.per_sample:
            for key in sample.metadata:
                if key not in seen_metadata:
                    seen_metadata.add(key)
                    metadata_fields.append(key)

        with _atomic_write(out_dir / "predictions.csv", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=base_fields + metadata_fields)
            writer.writeheader()
            for sample in result.per_sample:
                row = {
                    "sample_id": sample.sample_id,
                    "ground_truth": sample.ground_truth,
                    "prediction": sample.parsed_prediction,
                    "raw_prediction": sample.raw_prediction,
                    "input_tokens": sample.input_tokens,
                    "output_tokens": sample.output_tokens,
                }
                row.update(sample.metadata)
                writer.writerow(row)

        # metrics.json goes last. Two files can't be swapped in as one unit, so
        # the ordering picks which way they may disagree: a crash in between
        # leaves metrics lagging the predictions, and since the resume check
        # reads metrics, the run looks *less* complete than it is. Re-running is
        # safe; trusting a stale-but-larger count is not.
        with _atomic_write(out_dir / "metrics.json") as f:
            json.dump(result.metrics, f, indent=2)

        print(f"Results written to: {out_dir}")
        return out_dir
