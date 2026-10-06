#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
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
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from models.base import BaseModel, ModelResponse
from models.local import _get_best_device
from timef.schema import Recording


def _peek_lora_flag(checkpoint_path: str) -> bool:
    """Read ``lora_enabled`` off a saved OpenTSLM-SP checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return bool(ckpt.get("lora_enabled", False))


def _assemble_left_padded(
    model: OpenTSLMSP, batch: list[dict]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(inputs_embeds, attention_mask)`` for a batch, left-padded.

    This mirrors ``OpenTSLMSP.pad_and_apply_batch`` — same text tokenization,
    same per-sample interleaving — and differs in two places, both of which
    matter only once the batch holds more than one sample:

    1. The final assembly left-pads instead of right-padding. Upstream's right
       padding is correct for ``compute_loss``, which reads logits at explicit
       label positions, but wrong for generation: HF takes next-token logits
       from the *last* position, and under right padding that position is a
       zero pad vector for every sample except the longest in the batch.
    2. Time series are encoded in groups of equal length rather than all padded
       to the batch maximum (see step 3).

    At batch size 1 neither divergence applies, so this function is identical
    to upstream there — ``scripts/check_batched_generate.py`` asserts exactly
    that, which guards the logic we copied against upstream drift.
    """
    device = model.device
    hidden_size = model.llm.config.hidden_size

    # 1) Gather every text span in the batch: pre_prompt, one per time series, post_prompt.
    all_texts: list[str] = []
    text_ptrs: list[tuple[int, int]] = []
    ts_counts: list[int] = []
    for sample in batch:
        start = len(all_texts)
        all_texts.append(sample["pre_prompt"])
        all_texts.extend(sample["time_series_text"])
        all_texts.append(sample["post_prompt"])
        text_ptrs.append((start, len(all_texts)))
        ts_counts.append(len(sample["time_series_text"]))

    # 2) Tokenize and embed them in one pass. The per-span slicing below takes
    #    the *first* mask.sum() tokens, so this tokenization must right-pad.
    tok = model.tokenizer(
        all_texts,
        return_tensors="pt",
        padding=True,
        padding_side="right",
        truncation=True,
    )
    input_ids = tok.input_ids.to(device, non_blocking=True)
    attn_mask = tok.attention_mask.to(device, non_blocking=True)
    text_embeds = model.llm.get_input_embeddings()(input_ids)

    # 3) Encode and project the time series, grouped by length.
    #
    #    Upstream pads every series in the batch out to the batch's longest one
    #    before encoding. The encoder turns those zero pads into real patch
    #    embeddings, and the attention mask marks them as content, so a short
    #    series gets different embeddings depending on which samples happen to
    #    share its batch — batch composition leaks into predictions. MHC weekly
    #    data is uniformly 168 steps per channel so this never bites here, but
    #    encoding each distinct length on its own makes batching exact for any
    #    dataset and costs nothing when (as here) there is only one group.
    ts_list: list[torch.Tensor] = []
    for sample in batch:
        for ts in sample["time_series"]:
            ts_list.append(ts.unsqueeze(-1) if ts.dim() == 1 else ts)

    if ts_list:
        by_length: dict[int, list[int]] = {}
        for i, ts in enumerate(ts_list):
            by_length.setdefault(ts.size(0), []).append(i)

        projected: list[torch.Tensor | None] = [None] * len(ts_list)
        for length, idxs in by_length.items():
            group = torch.stack([ts_list[i] for i in idxs]).to(device, non_blocking=True)
            remainder = length % model.patch_size
            if remainder:
                pad = group.new_zeros(
                    group.size(0), model.patch_size - remainder, group.size(2)
                )
                group = torch.cat([group, pad], dim=1)
            group_proj = model.projector(model.encoder(group.squeeze(-1)))
            group_proj = group_proj.to(text_embeds.dtype)
            for slot, i in enumerate(idxs):
                projected[i] = group_proj[slot]
    else:
        projected = []

    # 4) Re-interleave each sample: pre_prompt, then (text, series) pairs, then post_prompt.
    seq_embeds: list[torch.Tensor] = []
    seq_masks: list[torch.Tensor] = []
    ts_offset = 0
    for (start, end), n_ts in zip(text_ptrs, ts_counts, strict=True):
        sample_embeds = text_embeds[start:end]
        sample_masks = attn_mask[start:end]
        parts_e: list[torch.Tensor] = []
        parts_m: list[torch.Tensor] = []

        length = int(sample_masks[0].sum())
        parts_e.append(sample_embeds[0, :length, :])
        parts_m.append(sample_masks[0, :length])

        for i in range(n_ts):
            idx = 1 + i
            length = int(sample_masks[idx].sum())
            parts_e.append(sample_embeds[idx, :length, :])
            parts_m.append(sample_masks[idx, :length])

            proj = projected[ts_offset + i]
            parts_e.append(proj)
            parts_m.append(torch.ones(proj.size(0), device=device, dtype=torch.long))

        ts_offset += n_ts

        length = int(sample_masks[-1].sum())
        parts_e.append(sample_embeds[-1, :length, :])
        parts_m.append(sample_masks[-1, :length])

        seq_embeds.append(torch.cat(parts_e, dim=0))
        seq_masks.append(torch.cat(parts_m, dim=0))

    # 5) Left-pad to a common length. This is the one place we diverge from upstream.
    batch_size = len(seq_embeds)
    max_len = max(seq.size(0) for seq in seq_embeds)
    inputs_embeds = seq_embeds[0].new_zeros(batch_size, max_len, hidden_size)
    attention_mask = seq_masks[0].new_zeros(batch_size, max_len)
    for i, (embeds, mask) in enumerate(zip(seq_embeds, seq_masks, strict=True)):
        inputs_embeds[i, max_len - embeds.size(0):, :] = embeds
        attention_mask[i, max_len - mask.size(0):] = mask

    return inputs_embeds, attention_mask


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
        self._model.eval()

    @torch.inference_mode()
    def generate_batch(
        self, samples: list[dict], recordings: list[Recording]
    ) -> list[ModelResponse]:
        """Generate for several samples in a single forward/decode pass.

        Roughly an order of magnitude faster per GPU than looping ``generate``,
        because the time-series encoder and the decode loop both run batched
        instead of once per sample. ``inference_mode`` matters here too:
        ``eval_prompt`` builds the encoder/projector graph under grad by
        default, which costs memory we'd rather spend on batch size.
        """
        if not samples:
            return []

        batch = [
            {
                "pre_prompt": sample["pre_prompt"],
                "post_prompt": sample["post_prompt"],
                "time_series": sample["time_series"],
                "time_series_text": sample["time_series_text"],
            }
            for sample in samples
        ]
        batch = extend_time_series_to_match_patch_size_and_aggregate(
            batch, normalize=self.config.normalize
        )

        inputs_embeds, attention_mask = _assemble_left_padded(self._model, batch)
        gen_ids = self._model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
        )
        texts = self._model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return [ModelResponse(text=text, input_tokens=0, output_tokens=0) for text in texts]

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
