#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
import json
import tempfile
from pathlib import Path

from datasets.features import features as ds_features
from datasets import load_from_disk
import numpy as np
import pyarrow.compute as pc
from torch.utils.data import Dataset

from mhc.constants import DATASET_DIR


if "List" not in ds_features._FEATURE_TYPES:
    ds_features._FEATURE_TYPES["List"] = ds_features.Sequence


class MHCDataset(Dataset):
    def __init__(self, min_wear_pct: float = 0.0, dataset_dir: str = DATASET_DIR):
        if not dataset_dir:
            raise ValueError("dataset_dir must not be empty")
        self.ds = self._load_dataset(dataset_dir)
        if min_wear_pct > 0:
            max_nonwear = 1440 * (1 - min_wear_pct / 100)
            mask = pc.less_equal(self.ds.data.column("total_nonwear_minutes"), max_nonwear)
            indices = [i for i, m in enumerate(mask.to_pylist()) if m]
            self.ds = self.ds.select(indices)
        # Expose the Array2D column directly as numpy — avoids Arrow→list→numpy roundtrip
        self.ds.set_format("numpy", columns=["values"], output_all_columns=True)

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

    @classmethod
    def _load_dataset(cls, dataset_dir: str):
        try:
            return load_from_disk(dataset_dir)
        except ValueError as exc:
            if "Feature type 'List' not found" not in str(exc):
                raise
            compat_dir = cls._build_compat_dataset_dir(Path(dataset_dir))
            return load_from_disk(str(compat_dir))

    @staticmethod
    def _build_compat_dataset_dir(dataset_dir: Path) -> Path:
        compat_dir = Path(tempfile.mkdtemp(prefix="mhc_hf_compat_"))

        for path in dataset_dir.iterdir():
            target = compat_dir / path.name
            if path.name == "dataset_info.json":
                payload = json.loads(path.read_text())
                payload["features"] = MHCDataset._rewrite_feature_types(payload["features"])
                target.write_text(json.dumps(payload), encoding="utf-8")
            else:
                target.symlink_to(path.resolve())

        return compat_dir

    @staticmethod
    def _rewrite_feature_types(value):
        if isinstance(value, dict):
            rewritten = {k: MHCDataset._rewrite_feature_types(v) for k, v in value.items()}
            if rewritten.get("_type") == "List":
                rewritten["_type"] = "Sequence"
            return rewritten
        if isinstance(value, list):
            return [MHCDataset._rewrite_feature_types(item) for item in value]
        return value
