#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Evaluator — orchestrates task, model, and dataset into results.

Responsibilities:
  - Iterate over samples (with optional cap).
  - Call model.generate() for each sample.
  - Parse predictions via the task.
  - Aggregate metrics at the end.
  - Return a structured EvalResult ready for the ResultsWriter.
"""
from __future__ import annotations

from dataclasses import dataclass

from tqdm import tqdm

from evaluation.tasks.base import EvalTask
from models.base import BaseModel, ModelResponse


@dataclass(slots=True)
class SampleResult:
    """Result for a single evaluated sample."""
    sample_id: str
    ground_truth: str
    raw_prediction: str
    parsed_prediction: str
    input_tokens: int
    output_tokens: int
    metadata: dict


@dataclass(slots=True)
class EvalResult:
    """Aggregated result for one model on one task."""
    model_name: str
    task_name: str
    per_sample: list[SampleResult]
    metrics: dict[str, float | int]


class Evaluator:
    """Runs a single (task, model, dataset) evaluation pass."""

    def run(
        self,
        task: EvalTask,
        model: BaseModel,
        dataset,
        model_name: str,
        max_samples: int | None = None,
    ) -> EvalResult:
        """Evaluate *model* on *dataset* using *task*.

        The dataset must be an OpenTSLM-style ``QADataset`` and expose
        ``get_eval_context(idx)`` returning ``(Recording, ground_truth)``.
        ``dataset[idx]`` provides the model-facing sample dict (format
        determined by ``model.format_sample_str``).
        """
        n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        per_sample: list[SampleResult] = []

        for i in tqdm(range(n), desc=f"{model_name} / {task.name}"):
            recording, ground_truth = dataset.get_eval_context(i)
            sample = dataset[i]

            response: ModelResponse = model.generate(sample=sample, recording=recording)
            parsed = task.parse_prediction(response.text)

            per_sample.append(SampleResult(
                sample_id=recording.row_id,
                ground_truth=ground_truth,
                raw_prediction=response.text,
                parsed_prediction=parsed,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                metadata={"user_id": recording.user_id, "date": recording.date},
            ))

        gts = [s.ground_truth for s in per_sample]
        preds = [s.parsed_prediction for s in per_sample]

        return EvalResult(
            model_name=model_name,
            task_name=task.name,
            per_sample=per_sample,
            metrics=task.aggregate_metrics(gts, preds),
        )
