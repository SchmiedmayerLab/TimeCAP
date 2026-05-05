#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
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
from timef.schema import Annotation, Recording, SignalView


def _get_best_device() -> str:
    """Pick CUDA > MPS > CPU based on availability (mirrors OpenTSLM)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class LocalConfig:
    model: str                       # "google/gemma-3-4b-it"
    device: str | None = None        # auto-detected if None (cuda > mps > cpu)
    torch_dtype: str = "bfloat16"    # maps to torch.bfloat16 etc.
    max_tokens: int = 1024


class LocalModel(BaseModel):
    """VLM backend via an in-process HF transformers pipeline (image-text-to-text)."""

    def __init__(self, config: LocalConfig, channel_config: ChannelConfig):
        super().__init__(channel_config)
        self.config = config
        device = config.device or _get_best_device()
        self._pipe = pipeline(
            "image-text-to-text",
            model=config.model,
            device=device,
            dtype=getattr(torch, config.torch_dtype),
        )

    def generate(self, sample: dict, recording: Recording) -> ModelResponse:
        return self._send(
            text=f"{sample['pre_prompt']}\n\n{sample['post_prompt']}",
            signals=list(recording.iter_channels()),
        )

    def process(
        self,
        prompt: str,
        signals: list[SignalView],
        annotations: list[Annotation] | None = None,
        multi_channel: bool = False,
    ) -> ModelResponse:
        """Captioning-review helper: send plots + arbitrary prompt (+ caption)."""
        text = prompt
        if annotations:
            caption = "\n".join(a.text for a in annotations if a.text)
            if caption:
                text = f"{caption}\n\n{prompt}"
        return self._send(text=text, signals=signals)

    def _send(self, text: str, signals: list[SignalView]) -> ModelResponse:
        content: list[dict] = []
        for signal in signals:
            content.append({"type": "image", "image": self._render_signal(signal)})
        content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]
        output = self._pipe(text=messages, max_new_tokens=self.config.max_tokens)
        generated = output[0]["generated_text"][-1]["content"]

        return ModelResponse(text=generated, input_tokens=0, output_tokens=0)

    def _render_signal(self, signal: SignalView) -> Image.Image:
        """Plot signal and convert to PIL Image."""
        fig = self._plot_signal(signal)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
