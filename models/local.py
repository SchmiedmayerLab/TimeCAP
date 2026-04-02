#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import pipeline

from extractors import ChannelConfig
from models.base import BaseModel, ModelResponse
from timef.schema import Signal


@dataclass(frozen=True)
class LocalConfig:
    model: str                       # "google/gemma-3-4b-it"
    device: str = "cuda"             # "cuda", "mps", "cpu"
    torch_dtype: str = "bfloat16"    # maps to torch.bfloat16 etc.
    max_tokens: int = 1024


class LocalModel(BaseModel):
    """Runs a HF transformers pipeline in-process. No server needed."""

    def __init__(self, config: LocalConfig, channel_config: ChannelConfig):
        super().__init__(channel_config)
        self.config = config
        self._pipe = pipeline(
            "image-text-to-text",
            model=config.model,
            device=config.device,
            torch_dtype=getattr(torch, config.torch_dtype),
        )

    def _call(
        self,
        prompt: str,
        signals: list[Signal],
        caption: str | None,
        multi_channel: bool,
    ) -> ModelResponse:
        images = [self._render_signal(s) for s in signals]

        content: list[dict] = []
        for img in images:
            content.append({"type": "image", "image": img})
        text = prompt
        if caption:
            text = f"{caption}\n\n{prompt}"
        content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]
        output = self._pipe(text=messages, max_new_tokens=self.config.max_tokens)
        generated = output[0]["generated_text"][-1]["content"]

        return ModelResponse(text=generated, input_tokens=0, output_tokens=0)

    def _render_signal(self, signal: Signal) -> Image.Image:
        """Plot signal and convert to PIL Image."""
        fig = self._plot_signal(signal)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
