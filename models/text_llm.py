#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Text-only LLM backend.

Mirrors OpenTSLM's ``CommonEvaluator`` text path: the dataset stringifies
each channel's tensor via a Gruver-style tokenizer, the model receives
``sample["prompt"]`` verbatim, and ``recording`` is ignored.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import pipeline

from models.base import BaseModel, ModelResponse
from models.gruver import pick_formatter
from models.local import _get_best_device
from timef.schema import Recording


@dataclass(frozen=True)
class TextLLMConfig:
    model: str                       # "meta-llama/Llama-3.2-1B"
    device: str | None = None        # auto-detected if None (cuda > mps > cpu)
    torch_dtype: str = "bfloat16"
    max_tokens: int = 1024
    temperature: float = 0.1         # low, we want deterministic labels
    max_input_tokens: int = 8192     # truncate prompt; eager CPU attention scales O(N^2)


class TextLLMModel(BaseModel):
    """Text-only LLM via HF ``text-generation`` pipeline — no images, no plots."""

    format_sample_str = True

    def __init__(self, config: TextLLMConfig):
        super().__init__(channel_config=None)
        self.config = config
        device = config.device or _get_best_device()
        self._pipe = pipeline(
            "text-generation",
            model=config.model,
            device=device,
            dtype=getattr(torch, config.torch_dtype),
            temperature=config.temperature,
        )

    @property
    def time_series_formatter(self):
        return pick_formatter(self.config.model)

    def generate(self, sample: dict, recording: Recording) -> ModelResponse:
        output = self._pipe(
            sample["prompt"],
            max_new_tokens=self.config.max_tokens,
            return_full_text=False,
            do_sample=False,  # greedy — bfloat16 softmax on CPU can NaN otherwise
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        text = output[0]["generated_text"] if output else ""
        return ModelResponse(text=text, input_tokens=0, output_tokens=0)
