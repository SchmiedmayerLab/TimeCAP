#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Evaluator — orchestrates task, model, and dataset into results.

Responsibilities:
  - Iterate over samples (with optional cap).
  - Call model.process() for each sample.
  - Parse predictions via the task.
  - Aggregate metrics at the end.
  - Return a structured EvalResult ready for the ResultsWriter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tqdm import tqdm

from evaluation.tasks.base import EvalSample, EvalTask
from models.base import BaseModel, ModelResponse
from torch.utils.data import Dataset


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
    """Runs a single (task, model, dataset) evaluation pass.

    Usage::

        evaluator = Evaluator()
        result = evaluator.run(task, model, dataset, model_name="gpt-4o")
        ResultsWriter().write(result)
    """

    def run(
        self,
        task: EvalTask,
        model: BaseModel,
        dataset: Dataset,
        model_name: str,
        max_samples: int | None = None,
    ) -> EvalResult:
        """Evaluate *model* on *dataset* using *task*.

        Args:
            task:        EvalTask instance (prompt building + metric aggregation).
            model:       BaseModel instance (ClientModel or LocalModel).
            dataset:     PyTorch Dataset yielding EvalSample objects.
            model_name:  Display name for the model (used in result paths).
            max_samples: If set, only the first N samples are evaluated.

        Returns:
            EvalResult with per-sample predictions and aggregated metrics.
        """
        n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        per_sample: list[SampleResult] = []
        n_errors = 0

        for i in tqdm(range(n), desc=f"{model_name} / {task.name}"):
            sample: EvalSample = dataset[i]
            prompt = task.build_prompt(sample)
            signals = list(sample.recording.iter_channels())

            try:
                response: ModelResponse = model.process(prompt=prompt, signals=signals)
                parsed = task.parse_prediction(response.text)
            except Exception as exc:
                print(f"\n[EvalRunner] Error on {sample.sample_id}: {exc}")
                response = ModelResponse(text="", input_tokens=0, output_tokens=0)
                parsed = "unknown"
                n_errors += 1

            per_sample.append(SampleResult(
                sample_id=sample.sample_id,
                ground_truth=sample.ground_truth,
                raw_prediction=response.text,
                parsed_prediction=parsed,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                metadata=sample.metadata,
            ))

        if n_errors:
            print(f"[Evaluator] {n_errors}/{n} samples failed (marked 'unknown').")

        gts = [s.ground_truth for s in per_sample]
        preds = [s.parsed_prediction for s in per_sample]
        metrics = task.aggregate_metrics(gts, preds)

        return EvalResult(
            model_name=model_name,
            task_name=task.name,
            per_sample=per_sample,
            metrics=metrics,
        )
