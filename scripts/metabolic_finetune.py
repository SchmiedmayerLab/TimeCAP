#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Stage-2 multi-task AR fine-tune on MHC downstream labels.

Loads a stage-1 captioning checkpoint and fine-tunes encoder + projector + LoRA
on a mix of (sample, target_label) pairs across multiple labels. Each sample's
post_prompt asks the model to pick the canonical answer string from a fixed
list (e.g. ``"Male"``/``"Female"``); loss is next-token CE on that
bare canonical-string target. Run ``evaluation/run_eval.py`` against the
saved checkpoint to compute downstream metrics.

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
    _iter_shard_rows,
    _load_split_assignment,
)
from time_series_datasets.mhc_label_lookup import (
    LABEL_TYPES,
    LabelLookup,
    decode_labels,
    build_metabolic_post_prompt,
)


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
    labels_df = decode_labels(labels_df, labels_df.columns)
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

    def __init__(self, split, eos, labels, export_dir, label_lookup_path, split_file):
        for label in labels:
            if label not in LABEL_TYPES:
                raise ValueError(f"Unknown label: {label!r}")

        user_to_bucket = _load_split_assignment(split_file)
        all_samples = _load_base_samples(export_dir, label_lookup_path, labels)
        split_samples = [
            s for s in all_samples if user_to_bucket.get(s["user_id"]) == split
        ]

        items: list[dict] = []
        pre_text = "You are an expert in wearable sensor time series analysis."
        for sample in split_samples:
            ts_prompts = self._build_ts_prompts(sample)
            for label in labels:
                value = sample["labels"].get(label)
                if value is None or pd.isna(value):
                    continue
                answer = str(value)
                if not answer.endswith(eos):
                    answer += eos
                d = PromptWithAnswer(
                    TextPrompt(pre_text.strip()),
                    ts_prompts,
                    TextPrompt(build_metabolic_post_prompt(label).strip()),
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


def _build_mixed(split, labels, eos, export_dir, label_lookup_path, split_file):
    ds = MHCMultiLabelQADataset(split, eos, labels, export_dir, label_lookup_path, split_file)
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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, help="Stage-1 checkpoint (latest.pt or best_model.pt)"
    )
    parser.add_argument("--label_lookup", required=True)
    parser.add_argument(
        "--split_file", required=True,
        help="MHC-benchmark canonical split JSON (sharable_users_seed42_2026.json).",
    )
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

    # Data
    eos = model.get_eos_token()
    train_ds = _build_mixed(
        "train", args.labels, eos, args.export_dir, args.label_lookup, args.split_file,
    )
    val_ds = _build_mixed(
        "validation", args.labels, eos, args.export_dir, args.label_lookup, args.split_file,
    )
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


if __name__ == "__main__":
    main()
