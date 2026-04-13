#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Metabolic prediction evaluation dataset over MHC weekly recordings.

Each sample is one week of hourly-aggregated sensor data from a single user,
paired with a static survey label (e.g. BiologicalSex) from MHC questionnaire
responses.

Label source:
  labels_dev_subset.parquet  (1 row per user, static survey labels)

User-level train/val/test splits use the same SHA-1 bucketing as
MHCCaptionQADataset and MHCActivityDataset to prevent cross-task leakage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from torch.utils.data import Dataset

from evaluation.tasks.base import EvalSample
from mhc_weekly.dataset import WeeklyMHCDataset
from mhc_weekly.transformer import WeeklyMHCTransformer
from time_series_datasets.mhc_label_lookup import LabelLookup, assign_split as _assign_split
from timef.schema import Annotation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WEEKLY_DIR = "data/mhc_dev_subset_weekly_hf"
DEFAULT_LABELS_PATH = "data/labels_dev_subset.parquet"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MHCMetabolicDataset(Dataset):
    """PyTorch Dataset of one-week sensor summaries for metabolic prediction.

    Each sample is one user-week of hourly sensor data paired with a static
    survey label from the MHC questionnaire.

    Args:
        label_column:    Column name in labels parquet to use as ground truth.
        label_values:    Canonical values to keep (others discarded).
                         Values are matched after string normalization so both
                         parquet booleans and string labels work.
                         E.g. ["Male", "Female"] for BiologicalSex.
        split:           "train", "validation", or "test".
        dataset_dir:     Path to the pre-built weekly HuggingFace dataset.
        labels_path:     Path to the labels parquet file.
        max_users:       Cap on users included (useful for development).
    """

    def __init__(
        self,
        label_column: str,
        label_values: list[str],
        split: Literal["train", "validation", "test"],
        dataset_dir: str | Path = DEFAULT_WEEKLY_DIR,
        labels_path: str | Path = DEFAULT_LABELS_PATH,
        max_users: int | None = None,
        captions_dir: str | Path | None = None,
    ) -> None:
        self._split = split
        self._label_column = label_column
        self._transformer = WeeklyMHCTransformer()

        # Optional: pre-generated captions from the captionizer pipeline.
        # When set, each sample's recording.annotations will contain the caption text,
        # which is forwarded to the model alongside the sensor plots.
        # TODO: replace captions_dir with the trained captionizer output once available.
        self._captions: dict[str, str] = {}
        if captions_dir is not None:
            self._captions = self._load_captions(captions_dir)
            print(f"  Loaded {len(self._captions)} captions from {captions_dir}")

        labels = self._load_labels(labels_path, label_column, label_values)

        weekly_ds = WeeklyMHCDataset(dataset_dir=str(dataset_dir))
        self._samples = self._build_index(weekly_ds, labels, max_users)

        print(
            f"MHCMetabolicDataset ({split}, {label_column}): "
            f"{len(self._samples)} samples"
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        row, ground_truth = self._samples[idx]
        recording = self._transformer.transform_row(row)

        # Attach caption as annotation if available.
        caption_key = f"{row['user_id']}:{row['week_start']}"
        if caption_key in self._captions:
            recording.annotations.append(Annotation(
                caption_type="weekly_caption",
                text=self._captions[caption_key],
            ))

        return EvalSample(
            sample_id=f"{row['user_id']}:week:{row['week_start']}:{self._label_column}",
            recording=recording,
            ground_truth=ground_truth,
            metadata={
                "user_id": row["user_id"],
                "week_start": row["week_start"],
                "label_column": self._label_column,
                "ground_truth": ground_truth,
                "n_valid_days": row["n_valid_days"],
                "n_valid_hours": row["n_valid_hours"],
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_captions(self, captions_dir: str | Path) -> dict[str, str]:
        """Load pre-generated captions from a parquet file.

        Expects a parquet with columns: user_id, week_start, caption.
        Key format: "{user_id}:{week_start}".

        TODO: once the captionizer pipeline is trained, replace the placeholder
        parquet in evaluation/generate_captions_placeholder.py with the real output.
        """
        path = Path(captions_dir) / "captions.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Captions file not found: {path}\n"
                "Run evaluation/generate_captions_placeholder.py to create a placeholder, "
                "or point --captions-dir at the trained captionizer output."
            )
        df = pd.read_parquet(path, columns=["user_id", "week_start", "caption"])
        return {
            f"{row.user_id}:{row.week_start}": row.caption
            for row in df.itertuples(index=False)
        }

    def _load_labels(
        self,
        labels_path: str | Path,
        label_column: str,
        label_values: list[str],
    ) -> dict[str, str]:
        """Load {user_id: label_str} from a user-level labels parquet.

        Uses LabelLookup.join_onlabels — the user-level join that requires no
        date column (as opposed to join_ondate which expects week_start/date).
        """
        df = LabelLookup(labels_path).join_onlabels(labels=[label_column])
        df = df[df[label_column].notna()].copy()
        df["_label_str"] = df[label_column].map(str)
        allowed = {str(value) for value in label_values}
        df = df[df["_label_str"].isin(allowed)]
        return dict(zip(df["user_id"], df["_label_str"]))

    def _build_index(
        self,
        weekly_ds: WeeklyMHCDataset,
        labels: dict[str, str],
        max_users: int | None,
    ) -> list[tuple[dict, str]]:
        """Filter weekly rows to those matching split + label, one per user."""
        seen_users: set[str] = set()
        samples: list[tuple[dict, str]] = []

        for row in weekly_ds:
            user_id = row["user_id"]
            if user_id not in labels:
                continue
            if _assign_split(user_id) != self._split:
                continue
            if user_id in seen_users:
                continue  # keep first week per user
            if max_users is not None and len(seen_users) >= max_users:
                break
            seen_users.add(user_id)
            samples.append((row, labels[user_id]))

        return samples
