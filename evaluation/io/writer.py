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
                    predictions.json  ← per-sample ground truth + prediction
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from evaluation.evaluator import EvalResult


class ResultsWriter:
    """Writes EvalResult files under results/<task>/<model>/<timestamp>/.

    Args:
        results_dir: Root directory for all evaluation outputs.
    """

    def __init__(self, results_dir: Path | str = Path("results")) -> None:
        self.results_dir = Path(results_dir)

    def write(self, result: EvalResult) -> Path:
        """Persist *result* to disk and return the output directory path.

        Creates:
            metrics.json     — aggregated metrics dict (directly JSON-serialisable).
            predictions.json — list of per-sample dicts.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Replace "/" in model names (e.g. "meta-llama/Llama-3.2-1B") with "_"
        safe_model = result.model_name.replace("/", "_")
        out_dir = self.results_dir / result.task_name / safe_model / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)

        # metrics.json
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(result.metrics, f, indent=2)

        # predictions.json
        predictions = [
            {
                "sample_id":          s.sample_id,
                "ground_truth":       s.ground_truth,
                "prediction":         s.parsed_prediction,
                "raw_prediction":     s.raw_prediction,
                "input_tokens":       s.input_tokens,
                "output_tokens":      s.output_tokens,
                "metadata":           s.metadata,
            }
            for s in result.per_sample
        ]
        with open(out_dir / "predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)

        print(f"Results written to: {out_dir}")
        return out_dir
