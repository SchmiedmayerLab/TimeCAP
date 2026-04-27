#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Multi-label metabolic QA dataset.

Emits one ``PromptWithAnswer`` item per (sample, target_label) pair where the
label value is non-NaN. Intentionally bypasses ``QADataset``'s class-level cache
that would otherwise make per-label instances share the first label's filtered
samples.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from opentslm.prompt.prompt_with_answer import PromptWithAnswer
from opentslm.prompt.text_prompt import TextPrompt

from mhc.constants import CONTINUOUS_CHANNELS
from time_series_datasets.mhc_base_qa_dataset import (
    MHCBaseQADataset,
    _assign_split,
    _iter_shard_rows,
)
from time_series_datasets.mhc_label_lookup import LABEL_TYPES, LabelLookup


DEFAULT_METABOLIC_LABELS = [
    "work",
    "BiologicalSex",
    "BMI_categories",
    "age",
    "sleep_diagnosis1",
    "Hypertension",
    "happiness_categories",
]


_BASE_CACHE: dict[tuple, list[dict]] = {}


def _load_base_samples(export_dir, label_lookup_path, labels):
    """Load arrow shards once + join the requested labels. Cached so that
    train/validation/test splits share the same load."""
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
    def __init__(self, split, EOS_TOKEN, labels, export_dir, label_lookup_path):
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
                if not answer.endswith(EOS_TOKEN):
                    answer += EOS_TOKEN
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
