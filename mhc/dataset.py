#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
import json

from datasets import load_from_disk
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from torch.utils.data import Dataset

from mhc.constants import DATASET_DIR


class MHCDataset(Dataset):
    def __init__(
        self,
        min_wear_pct: float = 0.0,
        dataset_dir: str = DATASET_DIR,
        split_file: str | None = None,
    ):
        """Load MHC daily dataset, optionally filtered by ``split_file``
        (MHC-benchmark's ``sharable_users_seed42_2026.json`` — rows with
        ``user_id`` outside train ∪ validation ∪ test are dropped)."""
        self.ds = load_from_disk(dataset_dir)

        if split_file is not None:
            with open(split_file) as f:
                splits = json.load(f)
            allowed_users = (
                set(splits["train"]) | set(splits["validation"]) | set(splits["test"])
            )
            mask = pc.is_in(
                self.ds.data.column("user_id"),
                value_set=pa.array(sorted(allowed_users)),
            )
            indices = [i for i, m in enumerate(mask.to_pylist()) if m]
            self.ds = self.ds.select(indices)

        if min_wear_pct > 0:
            max_nonwear = 1440 * (1 - min_wear_pct / 100)
            mask = pc.less_equal(self.ds.data.column("total_nonwear_minutes"), max_nonwear)
            indices = [i for i, m in enumerate(mask.to_pylist()) if m]
            self.ds = self.ds.select(indices)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        return self._parse(self.ds[idx])

    @staticmethod
    def _parse(row: dict) -> dict:
        return {
            "user_id": row["user_id"],
            "date": row["date"],
            "data": np.array(row["values"], dtype=np.float32),  # (19, 1440)
            "wear_pct": (1440 - row["total_nonwear_minutes"]) / 1440 * 100,
            "total_nonwear_minutes": float(row["total_nonwear_minutes"]),
            "has_any_data": list(row["has_any_data"]),
            "minutes_nonzero_or_nan": list(row["minutes_nonzero_or_nan"]),
            "channel_names": list(row["channel_names"]),
            "channel_units": list(row["channel_units"]),
            "channel_variance": list(row["channel_variance"]),
        }
