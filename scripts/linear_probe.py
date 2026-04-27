#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Linear-probe a trained OpenTSLMSP checkpoint against MHC downstream labels.

Two phases (first run does both; re-runs with --skip_extract skip phase 1):
    1. Extract a fixed-length embedding per sample (no gradients).
         --embedding_source projector  : mean-pool encoder+projector output  (fast, skips LLM, ignores LoRA)
         --embedding_source llm_last   : LLM forward, mean-pool last hidden state (captures LoRA + prompt context)
    2. Fit sklearn probes per label on the user-hash split, report metrics.

Example:
    python scripts/linear_probe.py \
        --checkpoint results/.../latest.pt \
        --label_lookup /scratch/.../weekly_labels_lookup.parquet \
        --export_dir exports/lean_smoke \
        --output_dir probe_results/stage1 \
        --embedding_source llm_last
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import torch
from torch.nn.utils.rnn import pad_sequence

from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

from mhc.constants import CONTINUOUS_CHANNELS
from time_series_datasets.mhc_label_lookup import LABEL_TYPES, LabelLookup


TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
PRE_PROMPT = "You are an expert in wearable sensor time series analysis."
POST_PROMPT = ""


def _assign_split(user_id: str) -> str:
    digest = int(hashlib.sha1(user_id.encode()).hexdigest(), 16)
    bucket = (digest % 1000) / 1000.0
    if bucket < TRAIN_FRAC:
        return "train"
    if bucket < TRAIN_FRAC + VAL_FRAC:
        return "validation"
    return "test"


def _normalize_channel(channel: np.ndarray, is_continuous: bool) -> tuple[np.ndarray, float, float]:
    if not is_continuous:
        return np.nan_to_num(channel, nan=0.0).astype(np.float32), 0.0, 1.0
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        return np.zeros_like(channel, dtype=np.float32), 0.0, 1.0
    mean = float(finite.mean())
    std = max(float(finite.std()), 1e-6)
    out = (channel - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), mean, std


def _build_sample(row: dict) -> dict:
    """Match MHCBaseQADataset's prompt + time-series formatting."""
    values = np.asarray(row["values"], dtype=np.float32)
    channel_names = row["channel_names"]
    display_names = row["display_names"]
    units = row["units"]

    texts: list[str] = []
    tensors: list[torch.Tensor] = []
    for i, name in enumerate(channel_names):
        is_cont = name in CONTINUOUS_CHANNELS
        normed, mean, std = _normalize_channel(values[i], is_cont)
        tensors.append(torch.from_numpy(normed))
        display = display_names[i]
        unit = units[i]
        unit_str = f" in {unit}" if unit else ""
        if is_cont:
            texts.append(
                f"The following is the {display}{unit_str}, "
                f"it has mean {mean:.4f} and std {std:.4f}:"
            )
        else:
            texts.append(
                f"The following is a per-minute binary mask for {display} "
                f"(1 = active, 0 = inactive, NaN = nonwear):"
            )
    return {
        "pre_prompt": PRE_PROMPT,
        "time_series_text": texts,
        "time_series": tensors,
        "post_prompt": POST_PROMPT,
    }


def _projector_embedding(model, batch_rows: list[dict]) -> torch.Tensor:
    n_ch = len(batch_rows[0]["channel_names"])
    ts_list: list[torch.Tensor] = []
    for row in batch_rows:
        values = np.asarray(row["values"], dtype=np.float32)
        channel_names = row["channel_names"]
        for i, name in enumerate(channel_names):
            normed, _, _ = _normalize_channel(values[i], name in CONTINUOUS_CHANNELS)
            ts_list.append(torch.from_numpy(normed))

    ts_padded = pad_sequence(ts_list, batch_first=True)
    T = ts_padded.size(1)
    rem = T % model.patch_size
    if rem:
        pad = ts_padded.new_zeros(ts_padded.size(0), model.patch_size - rem)
        ts_padded = torch.cat([ts_padded, pad], dim=1)
    ts_padded = ts_padded.to(model.device, non_blocking=True)

    enc = model.encoder(ts_padded)
    proj = model.projector(enc).float()
    B = len(batch_rows)
    return proj.view(B, n_ch, proj.size(1), proj.size(2)).mean(dim=(1, 2))


def _llm_last_embedding(model, batch_rows: list[dict]) -> torch.Tensor:
    samples = [_build_sample(row) for row in batch_rows]
    inputs_embeds, attention_mask = model.pad_and_apply_batch(samples)
    outputs = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1]  # [B, L, H]
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B, L, 1]
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled.float()


def _snapshot(path: Path, embeddings, user_ids, dates, completed, source) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        tmp,
        embeddings=embeddings[:completed],
        user_ids=np.array(user_ids[:completed]),
        dates=np.array(dates[:completed]),
        completed=np.array(completed),
        embedding_source=np.array(source),
    )
    tmp.replace(path)


def _finalize(out_path: Path, embeddings, user_ids, dates, completed, source, label_lookup_path) -> None:
    embeddings = embeddings[:completed]
    user_ids = user_ids[:completed]
    dates = dates[:completed]
    print(f"Joining labels for {completed} samples...")
    lookup = LabelLookup(label_lookup_path)
    labels_df = lookup.join(user_ids, dates)
    splits = np.array([_assign_split(uid) for uid in user_ids])
    np.savez_compressed(
        out_path / "embeddings.npz",
        embeddings=embeddings,
        user_ids=np.array(user_ids),
        dates=np.array(dates),
        splits=splits,
        embedding_source=np.array(source),
    )
    labels_df.to_parquet(out_path / "labels.parquet")
    # Clean up partial snapshot — final file supersedes it
    partial = out_path / "embeddings_partial.npz"
    if partial.exists():
        partial.unlink()
    print(f"Saved embeddings + labels to {out_path} (source={source})")


def extract_embeddings(args, model) -> None:
    export_dir = Path(args.export_dir)
    shards = sorted(export_dir.glob("recordings_*.arrow"))
    if not shards:
        raise FileNotFoundError(f"No shards in {export_dir}")

    rows = []
    for shard in shards:
        rows.extend(feather.read_table(shard).to_pylist())
    print(f"Loaded {len(rows)} rows from {len(shards)} shards")

    if args.max_samples and args.max_samples < len(rows):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(rows), size=args.max_samples, replace=False)
        rows = [rows[i] for i in sorted(idx)]
        print(f"Subsampled to {len(rows)} rows (seed={args.seed})")

    hidden = model.llm.config.hidden_size
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    partial_path = out_path / "embeddings_partial.npz"

    embeddings = np.zeros((len(rows), hidden), dtype=np.float32)
    user_ids: list[str] = [""] * len(rows)
    dates: list[str] = [""] * len(rows)
    completed = 0

    resume_from_path = None
    if partial_path.exists() and not args.no_resume:
        resume_from_path = partial_path
    elif (out_path / "embeddings.npz").exists() and not args.no_resume:
        resume_from_path = out_path / "embeddings.npz"
        print(f"Found finalized {resume_from_path.name}, treating as resume point.")

    if resume_from_path is not None:
        cached = np.load(resume_from_path, allow_pickle=True)
        prev_completed = int(cached["completed"]) if "completed" in cached.files else len(cached["embeddings"])
        prev_source = str(cached["embedding_source"])
        if prev_source != args.embedding_source:
            print(
                f"Partial file has source={prev_source} but args.embedding_source={args.embedding_source}."
                " Ignoring and restarting. Use --no_resume to silence, or delete partial manually."
            )
        elif prev_completed > len(rows):
            print(
                f"Partial file has {prev_completed} samples but current run has only {len(rows)} rows."
                " Ignoring (likely different --max_samples/seed). Use --no_resume to silence."
            )
        else:
            embeddings[:prev_completed] = cached["embeddings"]
            prev_users = list(cached["user_ids"])
            prev_dates = list(cached["dates"])
            user_ids[:prev_completed] = prev_users
            dates[:prev_completed] = prev_dates
            completed = prev_completed
            print(f"Resuming from sample {completed}/{len(rows)}")

    model.eval()
    get_embed = _projector_embedding if args.embedding_source == "projector" else _llm_last_embedding

    try:
        with torch.no_grad():
            for start in range(completed, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                pooled = get_embed(model, batch)
                embeddings[start : start + len(batch)] = pooled.cpu().numpy()
                for i, row in enumerate(batch):
                    user_ids[start + i] = str(row["user_id"])
                    dates[start + i] = str(row["date"])
                completed = start + len(batch)

                if (start // args.batch_size) % 50 == 0:
                    print(f"  {completed}/{len(rows)}")

                if args.save_every > 0 and completed % args.save_every < args.batch_size:
                    _snapshot(
                        partial_path, embeddings, user_ids, dates, completed, args.embedding_source
                    )
    except KeyboardInterrupt:
        print(f"\nInterrupted at {completed}/{len(rows)}. Saving partial snapshot...")
        _snapshot(
            partial_path, embeddings, user_ids, dates, completed, args.embedding_source
        )
        print(f"Partial saved to {partial_path}. Re-run to resume or pass --skip_extract to probe partial.")
        if completed == 0:
            raise
        _finalize(
            out_path, embeddings, user_ids, dates, completed, args.embedding_source, args.label_lookup
        )
        return

    _finalize(
        out_path, embeddings, user_ids, dates, completed, args.embedding_source, args.label_lookup
    )


def run_probes(args) -> None:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        balanced_accuracy_score,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
    )

    out_path = Path(args.output_dir)
    final_path = out_path / "embeddings.npz"
    partial_path = out_path / "embeddings_partial.npz"
    if not final_path.exists() and partial_path.exists():
        print(f"No final embeddings.npz — finalizing from partial snapshot.")
        cached = np.load(partial_path, allow_pickle=True)
        completed = int(cached["completed"])
        embeddings = cached["embeddings"]
        user_ids = list(cached["user_ids"])
        dates = list(cached["dates"])
        source = str(cached["embedding_source"])
        _finalize(out_path, embeddings, user_ids, dates, completed, source, args.label_lookup)
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
        if ltype == "binary":
            ytr = y_raw[tr].astype(int)
            yte = y_raw[te].astype(int)
            clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xtr, ytr)
            prob = clf.predict_proba(Xte)[:, 1]
            pred = (prob > 0.5).astype(int)
            results[label] = {
                "type": ltype,
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "auroc": float(roc_auc_score(yte, prob)),
                "bal_acc": float(balanced_accuracy_score(yte, pred)),
            }
        elif ltype == "ordinal":
            ytr = y_raw[tr].astype(int)
            yte = y_raw[te].astype(int)
            clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xtr, ytr)
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label_lookup", required=True)
    parser.add_argument("--export_dir", default="exports/lean_smoke")
    parser.add_argument("--output_dir", default="probe_results")
    parser.add_argument("--llm_id", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--embedding_source",
        choices=["projector", "llm_last"],
        default="llm_last",
        help="projector: encoder+projector only (fast, no LoRA). "
             "llm_last: full LLM forward, mean-pool last hidden state.",
    )
    parser.add_argument(
        "--lora_r", type=int, default=16, help="Used only if checkpoint has lora_state"
    )
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="If >0, randomly subsample this many rows before extraction.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_every", type=int, default=500,
        help="Snapshot partial embeddings every N samples. 0 disables.",
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Ignore existing embeddings_partial.npz and start from scratch.",
    )
    parser.add_argument(
        "--skip_extract",
        action="store_true",
        help="Reuse cached embeddings.npz/labels.parquet from --output_dir",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not args.skip_extract:
        model = OpenTSLMSP(llm_id=args.llm_id, device=device)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.encoder.load_state_dict(ckpt["encoder_state"])
        model.projector.load_state_dict(ckpt["projector_state"])
        if "lora_state" in ckpt:
            model.enable_lora(
                lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0
            )
            model.load_lora_state_from_checkpoint(ckpt, allow_missing=True)
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
        extract_embeddings(args, model)

    run_probes(args)


if __name__ == "__main__":
    main()
