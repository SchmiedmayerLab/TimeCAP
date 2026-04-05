#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

from extractors import ChannelConfig
from models.base import BaseModel, ModelResponse
from timef.schema import Signal

load_dotenv(override=True)


@dataclass(frozen=True)
class ClientConfig:
    base_url: str                    # "https://api.openai.com/v1" or "http://localhost:11434/v1"
    model: str                       # "gpt-4o", "gemma3", "Qwen/Qwen2.5-VL-7B"
    max_tokens: int = 4096
    temperature: float = 0.7
    max_requests: int = 100


class ClientModel(BaseModel):
    """Calls an OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, LM Studio, etc.)."""

    def __init__(self, config: ClientConfig, channel_config: ChannelConfig):
        super().__init__(channel_config)
        self.config = config
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=api_key,
        )
        self._request_count = 0

    def _call(
        self,
        prompt: str,
        signals: list[Signal],
        caption: str | None,
        multi_channel: bool,
    ) -> ModelResponse:
        content: list[dict] = []
        for signal in signals:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._render_signal(signal)},
            })

        text = prompt
        if caption:
            text = f"{caption}\n\n{prompt}"
        content.append({"type": "text", "text": text})

        if self._request_count >= self.config.max_requests:
            raise RuntimeError(
                f"Request limit reached ({self.config.max_requests}). "
                "Increase max_requests in ClientConfig to continue."
            )
        self._request_count += 1

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        usage = response.usage
        return ModelResponse(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    def _render_signal(self, signal: Signal) -> str:
        """Plot signal and return as a base64 data URL."""
        fig = self._plot_signal(signal)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return f"data:image/png;base64,{b64}"
