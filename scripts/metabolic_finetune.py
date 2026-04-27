#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Stage-2 multi-task AR fine-tune on MHC downstream labels.

Loads a stage-1 captioning checkpoint and fine-tunes encoder + projector + LoRA
on a mix of (sample, target_label) pairs across multiple labels. Each sample's
post_prompt tells the model which label to predict; loss is next-token CE on the
answer token. Eval computes per-label metrics via logits over candidate answer
tokens (binary/ordinal) or generation+parse (continuous).

Example:
    python scripts/metabolic_finetune.py \
        --checkpoint results/.../stage1_captioning/checkpoints/latest.pt \
        --label_lookup /home/users/mrosenbl/MHC-benchmark/data/processed/weekly_labels_lookup.parquet \
        --export_dir exports/lean_smoke \
        --output_dir results/.../stage2_multitask \
        --labels work BiologicalSex BMI_categories age sleep_diagnosis1 Hypertension happiness_categories
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model_config import GRAD_CLIP_NORM, PATCH_SIZE
from opentslm.prompt.prompt_with_answer import PromptWithAnswer
from opentslm.prompt.text_prompt import TextPrompt
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

from mhc.constants import CONTINUOUS_CHANNELS
from time_series_datasets.mhc_base_qa_dataset import (
    MHCBaseQADataset,
    _assign_split,
    _iter_shard_rows,
)
from time_series_datasets.mhc_label_lookup import LABEL_TYPES, LabelLookup


DEFAULT_LABELS = [
    "work",
    "BiologicalSex",
    "BMI_categories",
    "age",
    "sleep_diagnosis1",
    "Hypertension",
    "happiness_categories",
]


# ---------------------------------------------------------------------------
# Data: multi-label dataset that bypasses QADataset's class-level cache
# ---------------------------------------------------------------------------

_BASE_CACHE: dict[tuple, list[dict]] = {}


def _load_base_samples(export_dir, label_lookup_path, labels):
    """Load shards once + join all required labels. Cached at module level so
    train/val/test share the same load."""
    key = (str(export_dir), str(label_lookup_path), tuple(sorted(labels)))
    if key in _BASE_CACHE:
        return _BASE_CACHE[key]

    samples: list[dict] = []
    for row in _iter_shard_rows(Path(export_dir)):
        texts = [ann["text"] for ann in row["annotations"] if ann["text"]]
        if not texts:
            continue
        samples.append(
            {
                "user_id": row["user_id"],
                "date": row["date"],
                "values": row["values"],
                "channel_names": row["channel_names"],
                "display_names": row["display_names"],
                "units": row["units"],
            }
        )

    lookup = LabelLookup(label_lookup_path)
    user_ids = [s["user_id"] for s in samples]
    dates = [s["date"] for s in samples]
    labels_df = lookup.join(user_ids, dates, labels=list(labels))
    for sample, (_, row) in zip(samples, labels_df.iterrows(), strict=True):
        sample["labels"] = row.to_dict()

    print(f"Loaded {len(samples)} base samples from {export_dir} (labels={list(labels)})")
    _BASE_CACHE[key] = samples
    return samples


class MHCMultiLabelQADataset(Dataset):
    """Emits one PromptWithAnswer dict per (sample, target_label) pair where the
    label is non-NaN.

    Bypasses QADataset's class-level cache (which would otherwise make all
    per-label MHCMetabolicQADataset instances share the FIRST label's filtered
    samples — a silent correctness bug for multi-task training).
    """

    def __init__(self, split, eos, labels, export_dir, label_lookup_path):
        for label in labels:
            if label not in LABEL_TYPES:
                raise ValueError(f"Unknown label: {label!r}")

        all_samples = _load_base_samples(export_dir, label_lookup_path, labels)
        split_samples = [s for s in all_samples if _assign_split(s["user_id"]) == split]

        items: list[dict] = []
        pre_text = "You are an expert in wearable sensor time series analysis."
        for sample in split_samples:
            ts_prompts = self._build_ts_prompts(sample)
            for label in labels:
                value = sample["labels"].get(label)
                if value is None or pd.isna(value):
                    continue
                ltype = LABEL_TYPES[label]
                answer = str(int(value)) if ltype in ("binary", "ordinal") else str(value)
                if not answer.endswith(eos):
                    answer += eos
                d = PromptWithAnswer(
                    TextPrompt(pre_text.strip()),
                    ts_prompts,
                    TextPrompt(
                        f"Based on the sensor data, predict the value of "
                        f"{label} ({ltype}).".strip()
                    ),
                    answer.strip(),
                ).to_dict()
                d["_target_label"] = label
                items.append(d)

        self.items = items
        self.per_label_counts = {
            label: sum(1 for d in items if d["_target_label"] == label)
            for label in labels
        }

    @staticmethod
    def _build_ts_prompts(sample):
        values = np.asarray(sample["values"], dtype=np.float32)
        prompts = []
        for idx, name in enumerate(sample["channel_names"]):
            channel = torch.from_numpy(values[idx])
            display = sample["display_names"][idx]
            unit = sample["units"][idx]
            unit_str = f" in {unit}" if unit else ""
            if name in CONTINUOUS_CHANNELS:
                prompts.append(MHCBaseQADataset._continuous_prompt(channel, display, unit_str))
            else:
                prompts.append(MHCBaseQADataset._binary_prompt(channel, display))
        return prompts

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _build_mixed(split, labels, eos, export_dir, label_lookup_path):
    ds = MHCMultiLabelQADataset(split, eos, labels, export_dir, label_lookup_path)
    if len(ds) == 0:
        raise RuntimeError(f"No usable samples in split={split} for any label")
    counts = ", ".join(f"{k}={v}" for k, v in ds.per_label_counts.items())
    print(f"  [{split}] total={len(ds)} | per-label: {counts}")
    return ds


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _atomic_torch_save(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _save_latest(out_path: Path, model, optimizer, scheduler, epoch, step):
    ckpt = {
        "epoch": epoch,
        "step": step,
        "encoder_state": model.encoder.state_dict(),
        "projector_state": model.projector.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    model.save_lora_state_to_checkpoint(ckpt)
    _atomic_torch_save(ckpt, out_path / "latest.pt")


def _save_best(out_path: Path, model, epoch, val_loss):
    ckpt = {
        "epoch": epoch,
        "val_loss": val_loss,
        "encoder_state": model.encoder.state_dict(),
        "projector_state": model.projector.state_dict(),
    }
    model.save_lora_state_to_checkpoint(ckpt)
    _atomic_torch_save(ckpt, out_path / "best_model.pt")


def _apply_weights(model, ckpt: dict):
    model.encoder.load_state_dict(ckpt["encoder_state"])
    model.projector.load_state_dict(ckpt["projector_state"])
    model.load_lora_state_from_checkpoint(ckpt, allow_missing=True)


# ---------------------------------------------------------------------------
# Per-label evaluation
# ---------------------------------------------------------------------------


def _candidate_token_ids(tokenizer, label: str):
    ltype = LABEL_TYPES[label]
    if ltype == "binary":
        return [
            tokenizer.encode("0", add_special_tokens=False)[0],
            tokenizer.encode("1", add_special_tokens=False)[0],
        ]
    if ltype == "ordinal":
        # Enough for ≤10-class ordinals; unused classes just get low prob.
        return [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
    return None


def _answer_prefix_embed(model):
    """Embed of any tokens the tokenizer prepends to a plain answer string.

    `compute_loss` calls `tokenizer(answers, ...)` which uses
    `add_special_tokens=True` by default, so on Llama-3 the answer is stored as
    [BOS, digit, EOS]. To query P(digit) we must mirror this: append the prefix
    embed (BOS) to the prompt before forwarding, then read the next-token logits
    at that position.
    """
    tokenizer = model.tokenizer
    test_ids = tokenizer("X", add_special_tokens=True).input_ids
    plain_ids = tokenizer("X", add_special_tokens=False).input_ids
    if len(test_ids) <= len(plain_ids):
        return None
    prefix_ids = list(test_ids[: len(test_ids) - len(plain_ids)])
    if not prefix_ids:
        return None
    ids = torch.tensor([prefix_ids], device=model.device)
    return model.llm.get_input_embeddings()(ids)  # [1, P, H]


@torch.no_grad()
def _logits_over_candidates(model, batch, candidate_ids, prefix_emb):
    inputs_embeds, attention_mask = model.pad_and_apply_batch(batch)

    if prefix_emb is not None:
        B = inputs_embeds.size(0)
        pe = prefix_emb.expand(B, -1, -1).to(inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, pe], dim=1)
        pe_mask = torch.ones(
            B, pe.size(1), device=attention_mask.device, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat([attention_mask, pe_mask], dim=1)

    out = model.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = out.logits if hasattr(out, "logits") else out[0]
    last_idx = attention_mask.sum(dim=1) - 1  # last attended position
    rows = torch.arange(logits.size(0), device=logits.device)
    gathered = logits[rows, last_idx]  # [B, V]
    cand_logits = gathered[:, candidate_ids]
    return torch.softmax(cand_logits.float(), dim=-1).cpu().numpy()


def _parse_continuous(text: str) -> float:
    text = text.strip().split("\n")[0].split()[0] if text.strip() else ""
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _strip_eos(s: str, eos: str) -> str:
    return s.replace(eos, "").strip()


@torch.no_grad()
def evaluate_per_label(model, args, out_path: Path):
    from sklearn.metrics import (
        balanced_accuracy_score,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
    )

    eos = model.get_eos_token()
    test_ds = MHCMultiLabelQADataset(
        split="test",
        eos=eos,
        labels=args.labels,
        export_dir=args.export_dir,
        label_lookup_path=args.label_lookup,
    )

    by_label: dict[str, list[dict]] = defaultdict(list)
    for item in test_ds.items:
        by_label[item["_target_label"]].append(item)

    prefix_emb = _answer_prefix_embed(model)
    tokenizer = model.tokenizer
    results: dict[str, dict] = {}

    for label, items in by_label.items():
        ltype = LABEL_TYPES[label]
        loader = DataLoader(items, batch_size=args.eval_batch_size, collate_fn=list)

        preds: list[float] = []
        probs: list[np.ndarray] = []
        gts: list[float] = []

        if ltype in ("binary", "ordinal"):
            candidate_ids = _candidate_token_ids(tokenizer, label)
            for batch in tqdm(loader, desc=f"Eval {label}", dynamic_ncols=True):
                extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=PATCH_SIZE
                )
                p = _logits_over_candidates(model, batch, candidate_ids, prefix_emb)
                probs.append(p)
                preds.extend(p.argmax(axis=1).tolist())
                gts.extend(int(_strip_eos(b["answer"], eos)) for b in batch)
            probs_arr = (
                np.concatenate(probs, axis=0) if probs else np.zeros((0, len(candidate_ids)))
            )
        else:  # continuous
            for batch in tqdm(loader, desc=f"Eval {label}", dynamic_ncols=True):
                extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=PATCH_SIZE
                )
                gen = model.generate(batch, max_new_tokens=16)
                for g, b in zip(gen, batch):
                    preds.append(_parse_continuous(g))
                    gts.append(float(_strip_eos(b["answer"], eos)))

        preds_arr = np.asarray(preds)
        gts_arr = np.asarray(gts)

        if ltype == "binary":
            valid = np.isin(gts_arr, [0, 1])
            if valid.sum() < 10:
                continue
            prob_pos = probs_arr[valid, 1]
            try:
                auroc = float(roc_auc_score(gts_arr[valid], prob_pos))
            except ValueError:
                auroc = float("nan")
            results[label] = {
                "type": ltype,
                "n_test": int(valid.sum()),
                "auroc": auroc,
                "bal_acc": float(
                    balanced_accuracy_score(gts_arr[valid], preds_arr[valid])
                ),
            }
        elif ltype == "ordinal":
            results[label] = {
                "type": ltype,
                "n_test": int(len(gts_arr)),
                "bal_acc": float(balanced_accuracy_score(gts_arr, preds_arr)),
                "mae": float(mean_absolute_error(gts_arr, preds_arr)),
            }
        else:
            valid = ~np.isnan(preds_arr)
            if valid.sum() < 10:
                continue
            results[label] = {
                "type": ltype,
                "n_test": int(valid.sum()),
                "mae": float(mean_absolute_error(gts_arr[valid], preds_arr[valid])),
                "r2": float(r2_score(gts_arr[valid], preds_arr[valid])),
            }

        r = results[label]
        summary = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in r.items()
            if k != "type"
        )
        print(f"  {label} ({r['type']}): {summary}")

    with open(out_path / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path / 'eval_results.json'}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, help="Stage-1 checkpoint (latest.pt or best_model.pt)"
    )
    parser.add_argument("--label_lookup", required=True)
    parser.add_argument("--export_dir", default="exports/lean_smoke")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--llm_id", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_frac", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=GRAD_CLIP_NORM)
    parser.add_argument("--save_every_n_steps", type=int, default=200)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA fine-tune (only encoder+projector update)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    for label in args.labels:
        if label not in LABEL_TYPES:
            raise ValueError(f"Unknown label: {label!r}. Valid: {sorted(LABEL_TYPES)}")

    # Build model + load stage-1 weights
    model = OpenTSLMSP(llm_id=args.llm_id, device=device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not args.no_lora:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0)
    if args.gradient_checkpointing:
        # PEFT-friendly order: enable_lora first, then gradient checkpointing.
        model.llm.gradient_checkpointing_enable()
        if hasattr(model.llm, "enable_input_require_grads"):
            model.llm.enable_input_require_grads()
    _apply_weights(model, ckpt)
    val_loss_msg = ckpt.get("val_loss")
    val_loss_msg = f"{val_loss_msg:.4f}" if isinstance(val_loss_msg, float) else "n/a"
    print(
        f"Loaded stage-1 checkpoint (epoch={ckpt.get('epoch', '?')}, val_loss={val_loss_msg})"
    )

    if args.eval_only:
        model.eval()
        evaluate_per_label(model, args, out_path)
        return

    # Data
    eos = model.get_eos_token()
    train_ds = _build_mixed("train", args.labels, eos, args.export_dir, args.label_lookup)
    val_ds = _build_mixed("validation", args.labels, eos, args.export_dir, args.label_lookup)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=list
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=list
    )
    print(f"Train size={len(train_ds)}, Val size={len(val_ds)}")

    # Optimizer
    param_groups = [
        {"params": list(model.encoder.parameters()), "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": list(model.projector.parameters()), "lr": args.lr, "weight_decay": args.weight_decay},
    ]
    if not args.no_lora and model.lora_enabled:
        lora_params = list(model.get_lora_parameters())
        if lora_params:
            param_groups.append(
                {"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay}
            )
    optimizer = AdamW(param_groups, lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_frac * total_steps), total_steps
    )

    # Resume
    start_epoch = 0
    start_step = 0
    latest_path = out_path / "latest.pt"
    if args.resume and latest_path.exists():
        st = torch.load(latest_path, map_location="cpu", weights_only=False)
        _apply_weights(model, st)
        optimizer.load_state_dict(st["optimizer_state"])
        scheduler.load_state_dict(st["scheduler_state"])
        start_epoch = st["epoch"]
        start_step = st["step"] + 1
        print(f"Resumed from epoch={start_epoch} step={start_step}")

    best_val = float("inf")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        step_skip = start_step if epoch == start_epoch else 0
        pbar = tqdm(
            total=len(train_loader),
            initial=step_skip,
            desc=f"Epoch {epoch}/{args.epochs - 1} train",
            dynamic_ncols=True,
        )
        running = 0.0
        seen = 0
        for step, batch in enumerate(train_loader):
            if step < step_skip:
                continue
            extend_time_series_to_match_patch_size_and_aggregate(
                batch, patch_size=PATCH_SIZE
            )
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), args.grad_clip
            )
            optimizer.step()
            scheduler.step()
            running += loss.item()
            seen += 1
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{running / seen:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
            if (step + 1) % args.save_every_n_steps == 0:
                _save_latest(out_path, model, optimizer, scheduler, epoch, step)
        pbar.close()
        start_step = 0

        # Val
        model.eval()
        vloss = 0.0
        vsteps = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} val", dynamic_ncols=True):
                extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=PATCH_SIZE
                )
                vloss += model.compute_loss(batch).item()
                vsteps += 1
        val_loss = vloss / max(vsteps, 1)
        train_loss = running / max(seen, 1)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        with open(out_path / "loss_history.txt", "a") as f:
            f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")

        if val_loss + 1e-4 < best_val:
            best_val = val_loss
            _save_best(out_path, model, epoch, val_loss)
            print(f"Saved best (val_loss={val_loss:.4f})")

        _save_latest(out_path, model, optimizer, scheduler, epoch, len(train_loader) - 1)

    # Load best for final eval
    best_path = out_path / "best_model.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        _apply_weights(model, best_ckpt)
        bv = best_ckpt.get("val_loss")
        bv = f"{bv:.4f}" if isinstance(bv, float) else "n/a"
        print(f"Loaded best_model.pt for final eval (val_loss={bv})")

    model.eval()
    evaluate_per_label(model, args, out_path)


if __name__ == "__main__":
    main()
