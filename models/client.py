#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from dataclasses import dataclass

from extractors import ChannelConfig
from models.base import BaseModel, ModelResponse
from timef.schema import Signal


@dataclass(frozen=True)
class ClientConfig:
    base_url: str                    # "https://api.openai.com/v1" or "http://localhost:11434/v1"
    model: str                       # "gpt-4o", "gemma3", "Qwen/Qwen2.5-VL-7B"
    api_key: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.7


class ClientModel(BaseModel):
    """Calls an OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, LM Studio, etc.)."""

    def __init__(self, config: ClientConfig, channel_config: ChannelConfig):
        super().__init__(channel_config)
        self.config = config

    def _call(
        self,
        prompt: str,
        signals: list[Signal],
        caption: str | None,
        multi_channel: bool,
    ) -> ModelResponse:
        raise NotImplementedError
