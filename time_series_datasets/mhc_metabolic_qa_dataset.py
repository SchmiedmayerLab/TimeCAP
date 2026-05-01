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
from time_series_datasets.mhc_label_lookup import (
    LABEL_TYPES,
    build_metabolic_post_prompt,
)


class MHCMetabolicQADataset(MHCBaseQADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        target_label: str,
        export_dir: str | Path = "exports",
        label_lookup_path: str | Path | None = None,
    ) -> None:
        if target_label not in LABEL_TYPES:
            raise ValueError(f"Unknown label: {target_label!r}")
        if label_lookup_path is None:
            raise ValueError("label_lookup_path is required for MHCMetabolicQADataset")
        self._target_label = target_label
        super().__init__(
            split, EOS_TOKEN, export_dir, label_lookup_path,
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
        return build_metabolic_post_prompt(self._target_label)
