#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Verify that batched OpenTSLM-SP generation matches one-at-a-time generation.

Batching eval is only a speedup if it is also a no-op on the numbers, and the
failure mode is silent: a wrongly assembled batch produces fluent, plausible,
wrong text. This pins the property down on a tiny randomly-initialised model,
so it runs in seconds on CPU with no GPU and no gated download.

    python scripts/check_batched_generate.py

The assertion is on next-token *logits*, not decoded tokens. A random-init
model has a near-uniform output distribution, so its argmax flips under
perturbations far below the level that would matter for a trained model —
tokens are too unstable to test against. Logits are the real property, and
they are checked against a tolerance scaled to the logit spread.

Checks:
  1. assembly parity — at batch size 1, ``_assemble_left_padded`` is identical
     to upstream ``pad_and_apply_batch``. Guards the logic we copied.
  2. batch invariance, uniform series lengths — the MHC case, where every
     channel is 168 steps.
  3. batch invariance, ragged series lengths — the general case, which only
     holds because we encode time series grouped by length.
  4. upstream contrast — informational: the same measurement through
     ``pad_and_apply_batch``, showing the divergence this module avoids.

Exit code is nonzero if 1-3 fail.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from models.opentslm_native import _assemble_left_padded

TOKENIZER_ID = "meta-llama/Llama-3.2-1B"

# Batched and unbatched GEMMs reduce in different orders, so exact equality is
# not the bar. Two floors, whichever is larger: 1% of the logit spread, and a
# few ULPs of the working dtype. The second matters because OpenTSLMSP runs the
# LLM in bfloat16, whose epsilon (2^-7) is itself larger than the first floor —
# the check therefore runs at float32 too, where the tolerance is tight enough
# to actually constrain the assembly logic.
REL_TOL = 0.01
ULP_TOL = 4


def set_tf32(enabled: bool) -> None:
    """Toggle TF32 for fp32 matmuls, across the old and new torch APIs.

    On an H100, TF32 gives fp32 matmuls ~10 bits of mantissa. That is fine for
    the model but breaks the float32 leg of this check, whose whole purpose is
    a tolerance tight enough to catch an assembly mistake — under TF32 the
    noise floor alone would blow past it and report a failure that isn't one.
    """
    precision = "tf32" if enabled else "ieee"
    try:  # torch >= 2.9
        torch.backends.cuda.matmul.fp32_precision = precision
    except (AttributeError, RuntimeError, ValueError):
        try:
            torch.backends.cuda.matmul.allow_tf32 = enabled
        except (AttributeError, RuntimeError):
            pass
    try:
        torch.backends.cudnn.allow_tf32 = enabled
    except (AttributeError, RuntimeError):
        pass


def build_tiny_model(tmpdir: Path, device: str):
    """A 2-layer Llama with the real tokenizer, saved locally so OpenTSLMSP can load it."""
    from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(tmpdir)
    tokenizer.save_pretrained(tmpdir)

    model = OpenTSLMSP(llm_id=str(tmpdir), device=device)
    model.eval()
    model.llm.generation_config.do_sample = False
    return model


def make_samples(n: int, *, series_len: int | None = 168) -> list[dict]:
    """Samples whose prompt lengths vary, so left padding is non-trivial.

    Real eval samples vary the same way: each channel's prompt embeds that
    channel's mean and std, which tokenize to different lengths per sample.

    ``series_len=None`` gives每 sample a different series length (the ragged
    case); an int gives every sample the same length (the MHC case, 168).
    """
    torch.manual_seed(1)
    samples = []
    for i in range(n):
        n_series = 2 + (i % 3)
        length = series_len if series_len is not None else 24 + 8 * i
        samples.append({
            "pre_prompt": "You are an expert in wearable sensor time series analysis.",
            "time_series_text": [
                f"The following is channel {j}, it has mean {12.3456 * (i + 1) * (j + 1):.4f} "
                f"and std {0.98765 * (j + 1):.4f}:"
                for j in range(n_series)
            ],
            "time_series": [torch.randn(length) for _ in range(n_series)],
            "post_prompt": "Answer with one of: True, False.",
        })
    return samples


def _last_logits(model, samples, assemble) -> torch.Tensor:
    """Next-token logits for each sample: the value generation actually reads."""
    batch = extend_time_series_to_match_patch_size_and_aggregate(
        [dict(sample) for sample in samples]
    )
    with torch.inference_mode():
        inputs_embeds, attention_mask = assemble(model, batch)
        out = model.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    return out.logits[:, -1, :].float()


def _left(model, batch):
    return _assemble_left_padded(model, batch)


def _upstream(model, batch):
    return model.pad_and_apply_batch(batch)


def check_assembly_parity(model, samples) -> bool:
    """At batch size 1 there is no padding and one length group, so both agree exactly."""
    ok = True
    for i, sample in enumerate(samples):
        batch = extend_time_series_to_match_patch_size_and_aggregate([dict(sample)])
        with torch.inference_mode():
            ours_e, ours_m = _assemble_left_padded(model, batch)
        batch = extend_time_series_to_match_patch_size_and_aggregate([dict(sample)])
        with torch.inference_mode():
            theirs_e, theirs_m = model.pad_and_apply_batch(batch)

        matched = (
            ours_e.shape == theirs_e.shape
            and torch.equal(ours_e, theirs_e)
            and torch.equal(ours_m, theirs_m)
        )
        if not matched:
            ok = False
            print(f"  sample {i}: MISMATCH  {tuple(ours_e.shape)} vs {tuple(theirs_e.shape)}")
    if ok:
        print(f"  all {len(samples)} samples: identical to upstream at batch size 1")
    return ok


def check_batch_invariance(
    model, samples, assemble, label: str, dtype: torch.dtype
) -> tuple[bool, float]:
    """Batched next-token logits must match the one-at-a-time result."""
    model.llm.to(dtype)
    single = torch.cat([_last_logits(model, [s], assemble) for s in samples], dim=0)
    batched = _last_logits(model, samples, assemble)

    scale = single.std().item()
    tol = max(
        REL_TOL * scale,
        ULP_TOL * torch.finfo(dtype).eps * single.abs().max().item(),
    )
    worst = max(
        (batched[i] - single[i]).abs().max().item() for i in range(len(samples))
    )
    ok = worst <= tol
    print(f"  {label:<34s} worst |Δlogit| = {worst:.3e}  "
          f"(tol {tol:.3e})  {'PASS' if ok else 'FAIL'}")
    return ok, worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on. Defaults to cuda when available — batch "
             "invariance is a property of the kernels, so on a GPU node this "
             "must exercise the GPU to mean anything.",
    )
    args = parser.parse_args()

    device_note = args.device
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device_note = f"{args.device} ({torch.cuda.get_device_name(0)})"
    print(f"device: {device_note}  |  torch {torch.__version__}")

    # Disable TF32 for the whole check. The float32 leg needs it off or its
    # tolerance is meaningless; the bfloat16 leg is unaffected in the LLM
    # (already bf16) and only gains precision in the fp32 encoder, which makes
    # the check stricter rather than looser.
    set_tf32(False)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_tiny_model(Path(tmpdir), args.device)

        uniform = make_samples(6, series_len=168)
        ragged = make_samples(6, series_len=None)

        print("\n[1] assembly parity at batch size 1")
        parity_ok = check_assembly_parity(model, uniform)

        print("\n[2] batch invariance, uniform series length (the MHC case)")
        uniform_ok = all([
            check_batch_invariance(model, uniform, _left, "float32", torch.float32)[0],
            check_batch_invariance(model, uniform, _left, "bfloat16 (production dtype)",
                                   torch.bfloat16)[0],
        ])

        print("\n[3] batch invariance, ragged series lengths (the general case)")
        ragged_ok = all([
            check_batch_invariance(model, ragged, _left, "float32", torch.float32)[0],
            check_batch_invariance(model, ragged, _left, "bfloat16 (production dtype)",
                                   torch.bfloat16)[0],
        ])

        print("\n[4] same measurement through upstream pad_and_apply_batch (informational)")
        _, up_uniform = check_batch_invariance(
            model, uniform, _upstream, "upstream, uniform", torch.bfloat16)
        _, up_ragged = check_batch_invariance(
            model, ragged, _upstream, "upstream, ragged", torch.bfloat16)
        print(f"  divergence avoided: {up_uniform:.3e} (uniform), {up_ragged:.3e} (ragged)")

    print("\n" + "=" * 64)
    print(f"[1] assembly parity vs upstream at bs=1 : {'PASS' if parity_ok else 'FAIL'}")
    print(f"[2] batch invariance, uniform lengths   : {'PASS' if uniform_ok else 'FAIL'}")
    print(f"[3] batch invariance, ragged lengths    : {'PASS' if ragged_ok else 'FAIL'}")
    print("=" * 64)
    return 0 if (parity_ok and uniform_ok and ragged_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
