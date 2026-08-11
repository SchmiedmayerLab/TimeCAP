#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""OpenTSLM native backend.

Wraps ``opentslm.model.llm.OpenTSLMSP``: takes the dataset's native
sample dict (pre_prompt, post_prompt, time_series, time_series_text),
rebuilds a ``FullPrompt``, and calls ``eval_prompt`` for inference.

Two loading paths are supported:

* ``repo_id`` set — pulls a published OpenTSLM checkpoint from HF Hub
  via ``OpenTSLM.load_pretrained`` (auto-detects backbone from the
  ``-sp``/``-flamingo`` suffix).
* ``llm_id`` + ``checkpoint`` set — instantiates ``OpenTSLMSP`` with
  an explicit backbone and loads local encoder/projector weights. This
  is the path we use for our own trained checkpoints on Sherlock scratch.

LoRA is auto-detected by reading the ``lora_enabled`` flag off the
checkpoint — ``OpenTSLMSP.save_lora_state_to_checkpoint`` always writes
it, so this is safe for every SP checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.prompt.full_prompt import FullPrompt
from opentslm.prompt.text_prompt import TextPrompt
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt

from models.base import BaseModel, ModelResponse
from models.local import _get_best_device
from timef.schema import Recording


def _peek_lora_flag(checkpoint_path: str) -> bool:
    """Read ``lora_enabled`` off a saved OpenTSLM-SP checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return bool(ckpt.get("lora_enabled", False))


@dataclass(frozen=True)
class OpenTSLMConfig:
    repo_id: str | None = None        # HF Hub id, e.g. "OpenTSLM/gemma-3-270m-pt-har-sp"
    llm_id: str | None = None         # explicit backbone, used with ``checkpoint``
    checkpoint: str | None = None     # local .pt file, used with ``llm_id``
    device: str | None = None
    max_new_tokens: int = 512
    normalize: bool = False

    def __post_init__(self) -> None:
        has_hub = self.repo_id is not None
        has_local = self.llm_id is not None and self.checkpoint is not None
        if has_hub == has_local:
            raise ValueError(
                "OpenTSLMConfig: provide exactly one of `repo_id` (HF Hub) "
                "or (`llm_id` + `checkpoint`) for a local file."
            )


class OpenTSLMModel(BaseModel):
    """Native OpenTSLM-SP inference — encodes time-series internally."""

    def __init__(self, config: OpenTSLMConfig):
        super().__init__(channel_config=None)
        self.config = config
        device = config.device or _get_best_device()

        if config.repo_id is not None:
            # Pre-download to peek the LoRA flag; HF cache means load_pretrained
            # reuses the same file, no double-download.
            ckpt_path = hf_hub_download(config.repo_id, "model_checkpoint.pt")
            self._model = OpenTSLM.load_pretrained(
                config.repo_id, device=device, enable_lora=_peek_lora_flag(ckpt_path)
            )
        else:
            self._model = OpenTSLMSP(llm_id=config.llm_id, device=device)
            if _peek_lora_flag(config.checkpoint):
                self._model.enable_lora()
            self._model.load_from_file(config.checkpoint)
            self._model.eval()

        # Greedy decoding: bfloat16 softmax on CPU can emit NaN probabilities,
        # which torch.multinomial rejects. Greedy skips sampling entirely.
        self._model.llm.generation_config.do_sample = False

    def generate(self, sample: dict, recording: Recording) -> ModelResponse:
        prompt = FullPrompt(
            pre_prompt=TextPrompt(sample["pre_prompt"]),
            text_time_series_prompt_list=[
                TextTimeSeriesPrompt(text, tensor)
                for text, tensor in zip(
                    sample["time_series_text"], sample["time_series"], strict=True
                )
            ],
            post_prompt=TextPrompt(sample["post_prompt"]),
        )
        text = self._model.eval_prompt(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            normalize=self.config.normalize,
        )
        return ModelResponse(text=text, input_tokens=0, output_tokens=0)
