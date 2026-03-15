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

from mhc.constants import CHANNEL_NAMES, DATASET_DIR


class MHCDataset(Dataset):
    def __init__(self, split: str = "train", min_wear_pct: float = 0.0, dataset_dir: str = DATASET_DIR):
        self.ds = load_from_disk(f"{dataset_dir}/{split}")
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
            "data": np.array(row["data"], dtype=np.float32),        # (19, 1440)
            "nonwear": np.array(row["nonwear_vector"], dtype=np.uint8),  # (1440,)
            "wear_pct": (1440 - row["total_nonwear_minutes"]) / 1440 * 100,
        }
