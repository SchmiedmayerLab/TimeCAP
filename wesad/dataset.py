#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from pathlib import Path

from datasets import load_from_disk
from torch.utils.data import Dataset

from wesad.constants import TIMEF_DATASET_DIR


class WESADDataset(Dataset):
    def __init__(self, dataset_dir: str = TIMEF_DATASET_DIR):
        dataset_path = Path(dataset_dir)
        self.raw_dir = None
        looks_like_raw = dataset_path.exists() and any(
            path.is_dir() and (path / f"{path.name}.pkl").exists() for path in dataset_path.iterdir()
        )
        if dataset_path.exists() and not looks_like_raw:
            self.ds = load_from_disk(dataset_dir)
        else:
            raw_default = dataset_path if looks_like_raw else Path("data/WESAD")
            if not raw_default.exists():
                raise FileNotFoundError(f"Stored WESAD timef dataset not found at {dataset_path}.")
            self.ds = None
            self.raw_dir = raw_default
            self.subject_dirs = sorted(
                path for path in raw_default.iterdir() if path.is_dir() and (path / f"{path.name}.pkl").exists()
            )
            if not self.subject_dirs:
                raise FileNotFoundError(
                    f"No WESAD subject folders found under raw dataset directory {raw_default}."
                )

    def __len__(self) -> int:
        return len(self.ds) if self.ds is not None else len(self.subject_dirs)

    def __getitem__(self, idx: int) -> dict:
        if self.ds is not None:
            return self._parse(self.ds[idx])
        subject_dir = self.subject_dirs[idx]
        return {"subject_dir": str(subject_dir)}

    @staticmethod
    def _parse(row: dict) -> dict:
        return {
            "user_id": row["user_id"],
            "date": row["date"],
            "session_unix_start": float(row["session_unix_start"]),
            "data": row["values"],
            "has_any_data": row["has_any_data"],
            "channel_names": list(row["channel_names"]),
            "channel_units": list(row["channel_units"]),
            "label_values": row["label_values"],
            "label_offsets_seconds": row["label_offsets_seconds"],
            "e4_ibi_offsets_seconds": row["e4_ibi_offsets_seconds"],
            "e4_ibi_values_seconds": row["e4_ibi_values_seconds"],
            "source_metadata_lines": list(row.get("source_metadata_lines", [])),
        }
