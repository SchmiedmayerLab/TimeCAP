#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Concatenate multiple QA datasets behind the (split, EOS_TOKEN, **kw) contract
used by curriculum_learning.py's ``StageConfig.dataset_class``."""
from __future__ import annotations

from torch.utils.data import Dataset


class JointQADataset(Dataset):
    """Concatenate any number of QA datasets that share the
    (split, EOS_TOKEN, **kw) constructor contract.

    ``sub_datasets`` is a list of (dataset_class, sub_kwargs) tuples. Each entry
    is instantiated with the shared split + EOS_TOKEN and its own kwargs; items
    are exposed as a flat concat. DataLoader shuffling then produces a random
    mix across tasks.
    """

    def __init__(self, split, EOS_TOKEN, sub_datasets):
        self._parts = [
            cls(split=split, EOS_TOKEN=EOS_TOKEN, **kw) for cls, kw in sub_datasets
        ]
        self._lens = [len(p) for p in self._parts]
        self._total = sum(self._lens)

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        for part, n in zip(self._parts, self._lens):
            if idx < n:
                return part[idx]
            idx -= n
        raise IndexError(idx)
