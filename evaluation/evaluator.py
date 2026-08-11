#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
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
from transformers import logging as hf_logging

from evaluation.tasks.base import EvalTask
from models.base import BaseModel, ModelResponse

# Silence the per-generate "Setting pad_token_id to eos_token_id" warning.
hf_logging.set_verbosity_error()


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
        save_every: int | None = None,
        results_writer=None,
        out_dir=None,
    ) -> EvalResult:
        """Evaluate *model* on *dataset* using *task*.

        The dataset must be an OpenTSLM-style ``QADataset`` and expose
        ``get_eval_context(idx)`` returning ``(Recording, ground_truth)``.
        ``dataset[idx]`` provides the model-facing sample dict.

        With ``save_every`` + ``results_writer`` + ``out_dir`` set, partial
        results are flushed every N samples and on KeyboardInterrupt.
        """
        n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        per_sample: list[SampleResult] = []

        def _build_partial() -> "EvalResult":
            gts = [s.ground_truth for s in per_sample]
            preds = [s.parsed_prediction for s in per_sample]
            metrics = task.aggregate_metrics(gts, preds) if per_sample else {}
            return EvalResult(
                model_name=model_name,
                task_name=task.name,
                per_sample=per_sample,
                metrics=metrics,
            )

        def _flush() -> None:
            if results_writer is not None and out_dir is not None and per_sample:
                results_writer.write(_build_partial(), out_dir=out_dir)

        try:
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

                if save_every and (i + 1) % save_every == 0:
                    _flush()
        except KeyboardInterrupt:
            print(f"\n[interrupted at sample {len(per_sample)}/{n}] saving partial results")
            _flush()
            raise

        return _build_partial()
