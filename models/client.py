#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
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
from timef.schema import Annotation, Recording, SignalView

load_dotenv(override=True)


@dataclass(frozen=True)
class ClientConfig:
    model: str                       # "gpt-4o", "gemma3", "Qwen/Qwen2.5-VL-7B"
    base_url: str | None = None      # None → OpenAI SDK reads OPENAI_BASE_URL env, else api.openai.com
    max_tokens: int = 4096
    temperature: float = 0.7
    max_requests: int = 100


class ClientModel(BaseModel):
    """VLM backend over an OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, LM Studio, …)."""

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

    def generate(self, sample: dict, recording: Recording) -> ModelResponse:
        content = self._build_content(
            text=f"{sample['pre_prompt']}\n\n{sample['post_prompt']}",
            signals=list(recording.iter_channels()),
        )
        return self._send(content)

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
        return self._send(self._build_content(text=text, signals=signals))

    def _build_content(self, text: str, signals: list[SignalView]) -> list[dict]:
        content: list[dict] = []
        for signal in signals:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._render_signal(signal)},
            })
        content.append({"type": "text", "text": text})
        return content

    def _send(self, content: list[dict]) -> ModelResponse:
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

    def _render_signal(self, signal: SignalView) -> str:
        """Plot signal and return as a base64 data URL."""
        fig = self._plot_signal(signal)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return f"data:image/png;base64,{b64}"
