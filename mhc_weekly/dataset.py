#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
import json

from datasets import load_from_disk
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from torch.utils.data import Dataset

from mhc_weekly.constants import WEEKLY_DATASET_DIR


class WeeklyMHCDataset(Dataset):
    def __init__(
        self,
        min_valid_hours: int = 0,
        dataset_dir: str = WEEKLY_DATASET_DIR,
        split_file: str | None = None,
    ):
        """Load weekly MHC dataset, optionally filtered to sharable users from split_file."""
        self.ds = load_from_disk(dataset_dir)

        if split_file is not None:
            with open(split_file) as f:
                splits = json.load(f)
            allowed_users = (
                set(splits["train"]) | set(splits["validation"]) | set(splits["test"])
            )
            if not allowed_users:
                self.ds = self.ds.select([])
            else:
                col = self.ds.data.column("user_id")
                mask = pc.is_in(col, value_set=pa.array(sorted(allowed_users), type=col.type))
                indices = [i for i, m in enumerate(mask.to_pylist()) if m]
                self.ds = self.ds.select(indices)

        if min_valid_hours > 0:
            self.ds = self.ds.filter(lambda row: row["n_valid_hours"] >= min_valid_hours)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        return self._parse(self.ds[idx])

    @staticmethod
    def _parse(row: dict) -> dict:
        values = np.array(row["values"], dtype=np.float32)
        if "mask" in row:
            mask = np.array(row["mask"], dtype=np.float32)
            data = values.T
            mask = mask.T
            data[mask == 1] = np.nan
        else:
            data = values
            mask = np.isnan(data).astype(np.float32)
        return {
            "user_id": row["user_id"],
            "week_start": row.get("week_start") or row.get("date"),
            "data": data,
            "mask": mask,
            "n_valid_days": row.get("n_valid_days", 0),
            "n_valid_hours": row.get("n_valid_hours", 0),
            "n_data_days": row.get("n_data_days", 0),
        }
