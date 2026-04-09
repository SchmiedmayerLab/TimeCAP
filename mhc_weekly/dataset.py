#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from datasets import load_from_disk
import numpy as np
from torch.utils.data import Dataset

from mhc_weekly.constants import WEEKLY_DATASET_DIR


class WeeklyMHCDataset(Dataset):
    def __init__(self, min_valid_hours: int = 0, dataset_dir: str = WEEKLY_DATASET_DIR):
        self.ds = load_from_disk(dataset_dir)
        if min_valid_hours > 0:
            self.ds = self.ds.filter(lambda row: row["n_valid_hours"] >= min_valid_hours)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        return self._parse(self.ds[idx])

    @staticmethod
    def _parse(row: dict) -> dict:
        values = np.array(row["values"], dtype=np.float32)    # (168, 19)
        mask = np.array(row["mask"], dtype=np.float32)        # (168, 19)

        data = values.T                                       # (19, 168)
        mask = mask.T                                         # (19, 168)

        data[mask == 1] = np.nan

        return {
            "user_id": row["user_id"],
            "week_start": row["week_start"],
            "data": data,                                     # (19, 168)
            "mask": mask,                                     # (19, 168)
            "n_valid_days": row["n_valid_days"],
            "n_valid_hours": row["n_valid_hours"],
            "n_data_days": row["n_data_days"],
        }
