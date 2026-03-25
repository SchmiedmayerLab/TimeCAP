#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from datasets import load_from_disk
import numpy as np
import pyarrow.compute as pc
from torch.utils.data import Dataset

from mhc.constants import DATASET_DIR


class MHCDataset(Dataset):
    def __init__(self, min_wear_pct: float = 0.0, dataset_dir: str = DATASET_DIR):
        self.ds = load_from_disk(dataset_dir)
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
