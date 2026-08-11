#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Base QA dataset for MHC Arrow shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Literal, Tuple

import numpy as np
import pyarrow.feather as feather
import torch

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset

from mhc.constants import CONTINUOUS_CHANNELS
from time_series_datasets.mhc_label_lookup import LabelLookup, decode_labels
from timef.schema import Recording


DEFAULT_EXPORT_DIR = "exports"


def _load_split_assignment(
    split_file: str | Path,
) -> dict[str, Literal["train", "validation", "test"]]:
    """Invert MHC-benchmark's split JSON into a user_id → bucket map."""
    with open(split_file) as f:
        splits = json.load(f)
    user_to_bucket: dict[str, Literal["train", "validation", "test"]] = {}
    for bucket in ("train", "validation", "test"):
        for uid in splits[bucket]:
            user_to_bucket[uid] = bucket
    return user_to_bucket


def _iter_shard_rows(export_dir: Path) -> Iterator[dict]:
    if not export_dir.exists():
        raise FileNotFoundError(
            f"Export dir {export_dir} does not exist. "
            f"Run the caption pipeline with exporters.lean.write_caption_result first."
        )
    shards = sorted(export_dir.rglob("recordings_*.arrow"))
    if not shards:
        raise FileNotFoundError(
            f"No recordings_*.arrow shards found in {export_dir} (searched recursively)."
        )
    for shard_path in shards:
        table = feather.read_table(shard_path)
        for row in table.to_pylist():
            yield row


class MHCBaseQADataset(QADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        export_dir: str | Path = DEFAULT_EXPORT_DIR,
        label_lookup_path: str | Path | None = None,
        required_labels: List[str] | None = None,
        split_file: str | Path | None = None,
    ) -> None:
        """Load and format QA samples from captioned MHC arrow shards.

        required_labels: label columns to fetch from the labels parquet.
            ``None`` pulls every column registered in ``LABEL_TYPES``.
        split_file: MHC-benchmark's ``sharable_users_seed42_2026.json``.
            Required; users absent from the JSON are dropped.
        """
        if split_file is None:
            raise ValueError(
                "split_file is required: pass MHC-benchmark's "
                "sharable_users_seed42_2026.json"
            )
        self._split_name = split
        self._export_dir = Path(export_dir)
        self._label_lookup = LabelLookup(label_lookup_path) if label_lookup_path else None
        self._required_labels = required_labels
        self._user_to_bucket = _load_split_assignment(split_file)
        super().__init__(split, EOS_TOKEN)
        # self._raw_samples is populated inside _load_splits.

    def _load_splits(self) -> Tuple[List[dict], List[dict], List[dict]]:
        train: List[dict] = []
        validation: List[dict] = []
        test: List[dict] = []

        all_samples: List[dict] = []
        for row in _iter_shard_rows(self._export_dir):
            texts = [ann["text"] for ann in row["annotations"] if ann["text"]]
            sample = {
                "user_id": row["user_id"],
                "date": row["date"],
                "values": np.asarray(row["values"], dtype=np.float32),
                "channel_names": row["channel_names"],
                "display_names": row["display_names"],
                "units": row["units"],
                # Kept on the sample dict for MHCCaptionQADataset (training target).
                # Eval datasets don't read it; OpenTSLM's _format_sample never sees it.
                "caption": " ".join(texts),
            }
            all_samples.append(sample)

        if self._label_lookup and all_samples:
            user_ids = [s["user_id"] for s in all_samples]
            dates = [s["date"] for s in all_samples]
            labels_df = self._label_lookup.join(
                user_ids, dates, labels=self._required_labels
            )
            labels_df = decode_labels(labels_df, labels_df.columns)
            for sample, (_, row) in zip(all_samples, labels_df.iterrows(), strict=True):
                sample["labels"] = row.to_dict()

        for sample in all_samples:
            bucket = self._user_to_bucket.get(sample["user_id"])
            if bucket is None:
                continue
            {"train": train, "validation": validation, "test": test}[bucket].append(sample)

        # Keep this instance's split for get_eval_context() — QADataset discards
        # the raw dicts after _format_sample; only the current split is needed.
        self._raw_samples = {
            "train": train, "validation": validation, "test": test,
        }[self._split_name]

        return train, validation, test

    def get_eval_context(self, idx: int) -> Tuple[Recording, str]:
        """Return (recording, ground_truth) for evaluation.

        The recording carries raw per-channel values so VLM backends can
        render plots. Prompt text comes from ``dataset[idx]`` via the
        standard OpenTSLM formatting path — this method adds nothing to it.
        """
        row = self._raw_samples[idx]
        values = np.asarray(row["values"], dtype=np.float32)
        recording = Recording(
            row_id=f"{row['user_id']}:{row['date']}",
            user_id=row["user_id"],
            date=row["date"],
            values=values,
            channel_names=tuple(row["channel_names"]),
            display_names=tuple(row["display_names"]),
            units=tuple(row["units"]),
            has_any_data=np.any(~np.isnan(values), axis=1),
            channel_variance=np.nan_to_num(np.nanvar(values, axis=1), nan=0.0),
        )
        return recording, self._get_answer(row)

    def _get_pre_prompt(self, _row) -> str:
        return "You are an expert in wearable sensor time series analysis."

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        values = np.asarray(row["values"], dtype=np.float32)
        channel_names: List[str] = row["channel_names"]
        display_names: List[str] = row["display_names"]
        units: List[str | None] = row["units"]

        prompts: List[TextTimeSeriesPrompt] = []
        for idx, name in enumerate(channel_names):
            channel = torch.from_numpy(values[idx])
            display = display_names[idx]
            unit = units[idx]
            unit_str = f" in {unit}" if unit else ""

            if name in CONTINUOUS_CHANNELS:
                prompts.append(self._continuous_prompt(channel, display, unit_str))
            else:
                prompts.append(self._binary_prompt(channel, display))
        return prompts

    @staticmethod
    def _continuous_prompt(
        channel: torch.Tensor, display: str, unit_str: str
    ) -> TextTimeSeriesPrompt:
        finite_mask = torch.isfinite(channel)
        if finite_mask.any():
            finite_vals = channel[finite_mask]
            mean = float(finite_vals.mean())
            std = max(float(finite_vals.std(correction=0)), 1e-6)
        else:
            mean, std = 0.0, 1.0

        normalized = (channel - mean) / std
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        text = (
            f"The following is the {display}{unit_str}, "
            f"it has mean {mean:.4f} and std {std:.4f}:"
        )
        return TextTimeSeriesPrompt(text, normalized.numpy())

    @staticmethod
    def _binary_prompt(channel: torch.Tensor, display: str) -> TextTimeSeriesPrompt:
        channel = torch.nan_to_num(channel, nan=0.0)
        text = (
            f"The following is a per-minute binary mask for {display} "
            f"(1 = active, 0 = inactive or nonwear):"
        )
        return TextTimeSeriesPrompt(text, channel.numpy())
