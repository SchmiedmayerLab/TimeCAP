# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Linear-probe a frozen VLM against MHC downstream labels.

Renders weekly sensor tensors as multi-channel plots, feeds them to a frozen
vision-language model (e.g. Gemma 4 2B), extracts the last hidden state at the
final non-padding token, and trains sklearn probes on the resulting embeddings.

Two phases (first run does both; re-runs with --skip_extract skip phase 1):
    1. Extract a fixed-length embedding per sample (no gradients).
    2. Fit sklearn probes per label on the user-hash split, report metrics.

Example:
    python scripts/linear_probe_vlm.py \
        --hf_dataset /scratch/.../weekly_hf \
        --label_lookup /scratch/.../weekly_labels_lookup.parquet \
        --output_dir probe_results/gemma4_2b \
        --model_name google/gemma-4-E2B-it
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from time_series_datasets.mhc_label_lookup import LABEL_TYPES, LabelLookup


TRAIN_FRAC = 0.8
VAL_FRAC = 0.1

# Channel layout -- matches the MHC weekly HF dataset (168 hours, 19 channels)
_CONTINUOUS = [
    (0, "#e41a1c", "iPhone step count", "steps/hr"),
    (1, "#377eb8", "iPhone distance", "m/hr"),
    (2, "#4daf4a", "flights climbed (iPhone)", "count/hr"),
    (3, "#984ea3", "Apple Watch step count", "steps/hr"),
    (4, "#ff7f00", "Apple Watch distance", "m/hr"),
    (5, "#a65628", "heart rate", "bpm"),
    (6, "#f781bf", "active energy", "cal/hr"),
]

_BINARY = [
    (7, "#4c72b0", "Asleep"),
    (8, "#dd8452", "InBed"),
    (9, "#e41a1c", "Walking"),
    (10, "#377eb8", "Cycling"),
    (11, "#4daf4a", "Running"),
    (12, "#984ea3", "Other"),
    (13, "#ff7f00", "Mixed Cardio"),
    (14, "#a65628", "Strength"),
    (15, "#f781bf", "Elliptical"),
    (16, "#999999", "HIIT"),
    (17, "#66c2a5", "Functional"),
    (18, "#fc8d62", "Yoga"),
]

_CHANNEL_LEGEND = """\
This image shows one week of wearable sensor data. The x-axis shows hours (0h-24h repeating over 7 days).
- Panel 1 (Activity): iPhone/Watch steps, distance, flights climbed
- Panel 2 (Physiology): Heart rate (bpm), active energy (cal/hr)
- Panel 3 (Sleep): Binary asleep/inbed indicators (filled area = asleep)
- Panel 4 (Workouts): Binary workout type indicators"""  # noqa: E501


def _assign_split(user_id: str) -> str:
    digest = int(hashlib.sha1(user_id.encode()).hexdigest(), 16)
    bucket = (digest % 1000) / 1000.0
    if bucket < TRAIN_FRAC:
        return "train"
    if bucket < TRAIN_FRAC + VAL_FRAC:
        return "validation"
    return "test"


def _render_weekly_tensor(values: np.ndarray, dpi: int = 100) -> Image.Image:
    """Render a (168, 19) weekly sensor tensor as a single-column plot.

    Top: 7 continuous channels as line plots.
    Bottom: active binary channels as horizontal strips.
    """
    values = values.copy()
    # Heart rate (ch 5) is stored as beats/sec -- convert to bpm
    values[:, 5] = values[:, 5] * 60

    hours = np.arange(168)
    n_cont = len(_CONTINUOUS)

    active_binary = []
    for ch_idx, color, name in _BINARY:
        data = values[:, ch_idx]
        valid = ~np.isnan(data)
        if (valid & (data > 0.5)).any():
            active_binary.append((ch_idx, color, name))
    n_bin = len(active_binary)

    heights = [3] * n_cont + [1] * max(n_bin, 1)
    total_rows = n_cont + max(n_bin, 1)

    fig, axes = plt.subplots(
        total_rows, 1,
        figsize=(14, 1.2 * n_cont + 0.4 * max(n_bin, 1)),
        gridspec_kw={"height_ratios": heights, "hspace": 0.35},
    )
    if total_rows == 1:
        axes = [axes]

    for row, (ch_idx, color, name, unit) in enumerate(_CONTINUOUS):
        ax = axes[row]
        data = values[:, ch_idx]
        ax.plot(hours, data, linewidth=0.8, color=color, alpha=0.9)
        if np.all(np.isnan(data)):
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
        pad = max((vmax - vmin) * 0.15, 0.5)
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.set_xlim(0, 167)
        for d in range(1, 7):
            ax.axvline(d * 24, color="gray", linewidth=0.3, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        ax.set_title(
            f"{name} ({unit})  [{vmin:.0f}\u2013{vmax:.0f}]",
            fontsize=6, color=color, fontweight="bold", loc="left", pad=2,
        )

    if n_bin > 0:
        for i, (ch_idx, color, name) in enumerate(active_binary):
            ax = axes[n_cont + i]
            data = values[:, ch_idx]
            valid = ~np.isnan(data)
            active = valid & (data > 0.5)
            ax.fill_between(hours, 0, 1, where=active, color=color, alpha=0.8)
            ax.set_xlim(0, 167)
            ax.set_ylim(0, 1)
            for d in range(1, 7):
                ax.axvline(d * 24, color="gray", linewidth=0.3, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            ax.set_title(name, fontsize=5, color=color, fontweight="bold", loc="left", pad=1)
    else:
        ax = axes[n_cont]
        ax.set_xlim(0, 167)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        ax.set_title("(no activity/sleep data)", fontsize=5, color="gray", loc="left", pad=1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _auto_model_class(model_name: str):
    """Pick the right AutoModel class based on model name."""
    if "gemma-4" in model_name.lower():
        from transformers import AutoModelForMultimodalLM
        return AutoModelForMultimodalLM
    from transformers import AutoModelForImageTextToText
    return AutoModelForImageTextToText


def _load_model(args):
    """Load the frozen VLM and processor."""
    from transformers import AutoProcessor

    AutoModelCls = _auto_model_class(args.model_name)
    dtype = getattr(torch, args.torch_dtype)

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": {"": args.device} if args.device != "auto" else "auto",
        "attn_implementation": "eager",
    }
    if args.quantization == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    print(f"Loading frozen model {args.model_name}...")
    model = AutoModelCls.from_pretrained(args.model_name, **model_kwargs)
    model.eval()

    processor_name = args.model_name
    if "-pt" in processor_name:
        processor_name = processor_name.replace("-pt", "-it")
    processor = AutoProcessor.from_pretrained(processor_name)

    return model, processor


def _snapshot(path: Path, embeddings, user_ids, week_starts, completed, model_name) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        tmp,
        embeddings=embeddings[:completed],
        user_ids=np.array(user_ids[:completed]),
        week_starts=np.array(week_starts[:completed]),
        completed=np.array(completed),
        model_name=np.array(model_name),
    )
    tmp.replace(path)


def _finalize(out_path, embeddings, user_ids, week_starts, completed, model_name, label_lookup_path):
    embeddings = embeddings[:completed]
    user_ids = user_ids[:completed]
    week_starts = week_starts[:completed]
    print(f"Joining labels for {completed} samples...")
    lookup = LabelLookup(label_lookup_path)
    labels_df = lookup.join(user_ids, week_starts)
    splits = np.array([_assign_split(uid) for uid in user_ids])
    np.savez_compressed(
        out_path / "embeddings.npz",
        embeddings=embeddings,
        user_ids=np.array(user_ids),
        week_starts=np.array(week_starts),
        splits=splits,
        model_name=np.array(model_name),
    )
    labels_df.to_parquet(out_path / "labels.parquet")
    partial = out_path / "embeddings_partial.npz"
    if partial.exists():
        partial.unlink()
    print(f"Saved embeddings + labels to {out_path} (model={model_name})")


def extract_embeddings(args, model, processor) -> None:
    import datasets as hf_ds

    print(f"Loading HF dataset from {args.hf_dataset}...")
    ds = hf_ds.load_from_disk(args.hf_dataset)
    n_total = len(ds)
    print(f"Loaded {n_total} weekly samples")

    if args.max_samples and args.max_samples < n_total:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n_total, size=args.max_samples, replace=False)
        ds = ds.select(sorted(idx))
        n_total = len(ds)
        print(f"Subsampled to {n_total} samples (seed={args.seed})")

    # Determine hidden size from model config
    config = model.config
    if hasattr(config, "text_config"):
        hidden = config.text_config.hidden_size
    elif hasattr(config, "hidden_size"):
        hidden = config.hidden_size
    else:
        raise ValueError("Cannot determine hidden_size from model config")
    print(f"Hidden size: {hidden}")

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    partial_path = out_path / "embeddings_partial.npz"

    embeddings = np.zeros((n_total, hidden), dtype=np.float32)
    user_ids: list[str] = [""] * n_total
    week_starts: list[str] = [""] * n_total
    completed = 0

    # Resume logic
    resume_from_path = None
    if partial_path.exists() and not args.no_resume:
        resume_from_path = partial_path
    elif (out_path / "embeddings.npz").exists() and not args.no_resume:
        resume_from_path = out_path / "embeddings.npz"
        print(f"Found finalized {resume_from_path.name}, treating as resume point.")

    if resume_from_path is not None:
        cached = np.load(resume_from_path, allow_pickle=True)
        prev_completed = int(cached["completed"]) if "completed" in cached.files else len(cached["embeddings"])
        prev_model = str(cached["model_name"]) if "model_name" in cached.files else ""
        if prev_model and prev_model != args.model_name:
            print(
                f"Partial file has model={prev_model} but args.model_name={args.model_name}. "
                "Ignoring and restarting."
            )
        elif prev_completed > n_total:
            print(
                f"Partial file has {prev_completed} samples but current run has only {n_total}. "
                "Ignoring (likely different --max_samples/seed)."
            )
        else:
            embeddings[:prev_completed] = cached["embeddings"]
            user_ids[:prev_completed] = list(cached["user_ids"])
            week_starts[:prev_completed] = list(cached["week_starts"])
            completed = prev_completed
            print(f"Resuming from sample {completed}/{n_total}")

    prompt_text = _CHANNEL_LEGEND + "\n\nDescribe this person's health patterns."

    try:
        with torch.no_grad():
            for start in range(completed, n_total, args.batch_size):
                end = min(start + args.batch_size, n_total)
                batch_texts = []
                batch_images = []

                for i in range(start, end):
                    row = ds[i]
                    values = np.array(row["values"], dtype=np.float32)
                    image = _render_weekly_tensor(values, dpi=args.rendering_dpi)
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ]}]
                    text = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False,
                    )
                    batch_texts.append(text.strip())
                    batch_images.append([image])

                    user_ids[i] = str(row["user_id"])
                    week_starts[i] = str(row["week_start"])

                inputs = processor(
                    text=batch_texts, images=batch_images,
                    return_tensors="pt", padding=True,
                ).to(model.device)

                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # (B, L, H)

                pad_id = processor.tokenizer.pad_token_id
                for b in range(hidden_states.shape[0]):
                    ids = inputs["input_ids"][b]
                    non_pad = (ids != pad_id).nonzero(as_tuple=True)[0]
                    last_pos = non_pad[-1].item() if len(non_pad) > 0 else hidden_states.shape[1] - 1
                    feat = hidden_states[b, last_pos].float().cpu().numpy()
                    embeddings[start + b] = feat

                completed = end

                if (start // args.batch_size) % 50 == 0:
                    print(f"  {completed}/{n_total}")

                if args.save_every > 0 and completed % args.save_every < args.batch_size:
                    _snapshot(partial_path, embeddings, user_ids, week_starts, completed, args.model_name)

    except KeyboardInterrupt:
        print(f"\nInterrupted at {completed}/{n_total}. Saving partial snapshot...")
        _snapshot(partial_path, embeddings, user_ids, week_starts, completed, args.model_name)
        print(f"Partial saved to {partial_path}. Re-run to resume or pass --skip_extract to probe partial.")
        if completed == 0:
            raise
        _finalize(out_path, embeddings, user_ids, week_starts, completed, args.model_name, args.label_lookup)
        return

    _finalize(out_path, embeddings, user_ids, week_starts, completed, args.model_name, args.label_lookup)


def run_probes(args) -> None:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        balanced_accuracy_score,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler

    out_path = Path(args.output_dir)
    final_path = out_path / "embeddings.npz"
    partial_path = out_path / "embeddings_partial.npz"
    if not final_path.exists() and partial_path.exists():
        print("No final embeddings.npz -- finalizing from partial snapshot.")
        cached = np.load(partial_path, allow_pickle=True)
        completed = int(cached["completed"])
        _finalize(
            out_path, cached["embeddings"], list(cached["user_ids"]),
            list(cached["week_starts"]), completed,
            str(cached["model_name"]), args.label_lookup,
        )
    cached = np.load(final_path, allow_pickle=True)
    X = cached["embeddings"]
    splits = cached["splits"]
    labels_df = pd.read_parquet(out_path / "labels.parquet")

    train_mask = splits == "train"
    test_mask = splits == "test"
    print(f"Splits: train={train_mask.sum()}, test={test_mask.sum()}")

    results: dict[str, dict] = {}
    for label in labels_df.columns:
        ltype = LABEL_TYPES.get(label)
        if ltype is None:
            continue
        y_raw = labels_df[label].to_numpy()
        valid = ~pd.isna(y_raw)
        tr = train_mask & valid
        te = test_mask & valid
        if tr.sum() < 50 or te.sum() < 50:
            continue

        Xtr, Xte = X[tr], X[te]
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
        if ltype in ("binary", "ordinal"):
            ytr = y_raw[tr].astype(int)
            yte = y_raw[te].astype(int)
            if len(np.unique(ytr)) < 2:
                print(f"  {label} ({ltype}): skip -- only one class in train")
                continue
            clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xtr, ytr)
            if ltype == "binary":
                prob = clf.predict_proba(Xte)[:, 1]
                pred = (prob > 0.5).astype(int)
                auroc = (
                    float(roc_auc_score(yte, prob))
                    if len(np.unique(yte)) >= 2
                    else float("nan")
                )
                results[label] = {
                    "type": ltype,
                    "n_train": int(tr.sum()),
                    "n_test": int(te.sum()),
                    "auroc": auroc,
                    "bal_acc": float(balanced_accuracy_score(yte, pred)),
                }
            else:  # ordinal
                pred = clf.predict(Xte)
                results[label] = {
                    "type": ltype,
                    "n_train": int(tr.sum()),
                    "n_test": int(te.sum()),
                    "bal_acc": float(balanced_accuracy_score(yte, pred)),
                    "mae": float(mean_absolute_error(yte, pred)),
                }
        else:  # continuous
            ytr = y_raw[tr].astype(float)
            yte = y_raw[te].astype(float)
            reg = Ridge(alpha=1.0).fit(Xtr, ytr)
            pred = reg.predict(Xte)
            results[label] = {
                "type": ltype,
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "mae": float(mean_absolute_error(yte, pred)),
                "r2": float(r2_score(yte, pred)),
            }

        r = results[label]
        summary = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in r.items()
            if k != "type"
        )
        print(f"  {label} ({r['type']}): {summary}")

    with open(out_path / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path / 'probe_results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", required=True, help="Path to weekly HF dataset on disk")
    parser.add_argument("--label_lookup", required=True, help="Path to weekly_labels_lookup.parquet")
    parser.add_argument("--output_dir", default="probe_results/vlm")
    parser.add_argument("--model_name", default="google/gemma-4-E2B-it")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:0', 'cpu'")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--quantization", default=None, choices=[None, "4bit"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--rendering_dpi", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=0, help="If >0, subsample this many rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--skip_extract", action="store_true", help="Reuse cached embeddings, only run probes")
    args = parser.parse_args()

    if not args.skip_extract:
        model, processor = _load_model(args)
        extract_embeddings(args, model, processor)
        del model, processor
        torch.cuda.empty_cache()

    run_probes(args)


if __name__ == "__main__":
    main()
