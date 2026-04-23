#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Metabolic health-label prediction QA dataset."""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Tuple

import pandas as pd

from time_series_datasets.mhc_base_qa_dataset import MHCBaseQADataset
from time_series_datasets.mhc_label_lookup import LABEL_TYPES, METABOLIC_LABEL_CONFIG


class MHCMetabolicQADataset(MHCBaseQADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        target_label: str,
        export_dir: str | Path = "exports",
        label_lookup_path: str | Path | None = None,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ) -> None:
        if target_label not in LABEL_TYPES:
            raise ValueError(f"Unknown label: {target_label!r}")
        if label_lookup_path is None:
            raise ValueError("label_lookup_path is required for MHCMetabolicQADataset")
        self._target_label = target_label
        super().__init__(
            split, EOS_TOKEN, export_dir, label_lookup_path,
            format_sample_str, time_series_format_function,
            required_labels=[target_label],
        )
        if self._target_label not in self._label_lookup.label_names:
            raise ValueError(
                f"Label {self._target_label!r} not found in {label_lookup_path}"
            )

    def _load_splits(self) -> Tuple[List[dict], List[dict], List[dict]]:
        train, validation, test = super()._load_splits()
        train = [s for s in train if not pd.isna(s["labels"][self._target_label])]
        validation = [s for s in validation if not pd.isna(s["labels"][self._target_label])]
        test = [s for s in test if not pd.isna(s["labels"][self._target_label])]
        # Overwrite with filtered version for this instance's split.
        self._raw_samples = {
            "train": train, "validation": validation, "test": test,
        }[self._split_name]
        return train, validation, test

    def _get_answer(self, row) -> str:
        return str(row["labels"][self._target_label])

    def _get_post_prompt(self, _row) -> str:
        cfg = METABOLIC_LABEL_CONFIG.get(self._target_label)
        if cfg is None:
            label_type = LABEL_TYPES[self._target_label]
            return (
                f"Based on the sensor data, predict the value of "
                f"{self._target_label} ({label_type})."
            )
        values_str = ", ".join(cfg["values"])
        return (
            f"{cfg['question']}\n\n"
            f"Possible answers: {values_str}\n\n"
            "Rules:\n"
            "- Base your answer on the sensor patterns in the plots and the "
            "observations above.\n"
            "- You MUST give a classification even if the signal is unclear — "
            "state limitations but still make your best guess.\n"
            "- Never respond with a question.\n"
            "- You MUST end your response with 'Answer: <label>'."
        )
