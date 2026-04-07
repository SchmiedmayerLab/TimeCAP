#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""OpenTSLM QA dataset over MHC caption shards from exporters/lean.py.

One sample per recording, answer = space-joined annotations of all types
(statistical, structural, semantic). User-level split prevents leakage
across train/val/test.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator, List, Literal, Tuple

import numpy as np
import pyarrow.feather as feather
import torch

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset

from mhc.constants import CONTINUOUS_CHANNELS


DEFAULT_EXPORT_DIR = "exports"
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1


def _assign_split(user_id: str) -> Literal["train", "validation", "test"]:
    digest = int(hashlib.sha1(user_id.encode()).hexdigest(), 16)
    bucket = (digest % 1000) / 1000.0
    if bucket < TRAIN_FRAC:
        return "train"
    if bucket < TRAIN_FRAC + VAL_FRAC:
        return "validation"
    return "test"


def _iter_shard_rows(export_dir: Path) -> Iterator[dict]:
    if not export_dir.exists():
        raise FileNotFoundError(
            f"Export dir {export_dir} does not exist. "
            f"Run the caption pipeline with exporters.lean.write_caption_result first."
        )
    shards = sorted(export_dir.glob("recordings_*.arrow"))
    if not shards:
        raise FileNotFoundError(
            f"No recordings_*.arrow shards found in {export_dir}."
        )
    for shard_path in shards:
        table = feather.read_table(shard_path)
        for row in table.to_pylist():
            yield row


class MHCCaptionQADataset(QADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        export_dir: str | Path = DEFAULT_EXPORT_DIR,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ) -> None:
        self._export_dir = Path(export_dir)
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[List[dict], List[dict], List[dict]]:
        train: List[dict] = []
        validation: List[dict] = []
        test: List[dict] = []

        for row in _iter_shard_rows(self._export_dir):
            texts = [ann["text"] for ann in row["annotations"] if ann["text"]]
            if not texts:
                continue
            bucket = _assign_split(row["user_id"])
            target = {"train": train, "validation": validation, "test": test}[bucket]

            caption = " ".join(texts)
            sample = {
                "user_id": row["user_id"],
                "date": row["date"],
                "values": row["values"],
                "channel_names": row["channel_names"],
                "display_names": row["display_names"],
                "units": row["units"],
                "caption": caption,
            }
            target.append(sample)

        return train, validation, test

    def _get_answer(self, row) -> str:
        return row["caption"]

    def _get_pre_prompt(self, _row) -> str:
        return "You are an expert in wearable sensor time series analysis."

    def _get_post_prompt(self, _row) -> str:
        return (
            "Please generate a detailed caption for this day of sensor data, "
            "describing the person's activity, sleep, and physiological "
            "patterns as accurately as possible."
        )

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

        # NaN→0 after normalization so prompt mean/std describe the real signal.
        normalized = (channel - mean) / std
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        text = (
            f"The following is the {display}{unit_str}, "
            f"it has mean {mean:.4f} and std {std:.4f}:"
        )
        return TextTimeSeriesPrompt(text, normalized.tolist())

    @staticmethod
    def _binary_prompt(channel: torch.Tensor, display: str) -> TextTimeSeriesPrompt:
        text = (
            f"The following is a per-minute binary mask for {display} "
            f"(1 = active, 0 = inactive, NaN = nonwear):"
        )
        return TextTimeSeriesPrompt(text, channel.tolist())


if __name__ == "__main__":
    train_dataset = MHCCaptionQADataset("train", "")
    val_dataset = MHCCaptionQADataset("validation", "")
    test_dataset = MHCCaptionQADataset("test", "")
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print("\nSample keys:", list(sample.keys()))
        print("Pre-prompt:", sample["pre_prompt"])
        print("Post-prompt:", sample["post_prompt"])
        print("Answer:", sample["answer"][:200], "...")
        print("Num time series prompts:", len(sample["time_series"]))
        if sample["time_series"]:
            print("First time series text:", sample["time_series_text"][0])
            print("First time series length:", len(sample["time_series"][0]))
