#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Split an eval dataset across processes, one shard per GPU."""
from __future__ import annotations


class ShardView:
    """Every ``num_shards``-th sample, offset by ``shard_index``.

    Strided rather than contiguous on purpose. Eval datasets here come out in
    acquisition order — WESAD windows follow the study protocol (a long baseline
    block, then stress, then meditation, then amusement), MIMIC stays are sorted
    by ``stay_id`` — so contiguous blocks would hand one shard a near-single-class
    subset. Its partial metrics would then be meaningless on their own, and a
    failed shard would drop one class entirely rather than thinning all of them.
    Striding keeps every shard roughly class-balanced.

    Merge the per-shard predictions.csv files to get dataset-level numbers
    either way — see ``scripts/merge_eval_shards.py``. Wraps any dataset
    exposing ``__len__``, ``__getitem__`` and ``get_eval_context``.
    """

    def __init__(self, dataset, num_shards: int, shard_index: int) -> None:
        if num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {num_shards}")
        if not 0 <= shard_index < num_shards:
            raise ValueError(
                f"shard_index must be in [0, {num_shards}), got {shard_index}"
            )
        self._dataset = dataset
        self._idxs = list(range(shard_index, len(dataset), num_shards))

    def __len__(self) -> int:
        return len(self._idxs)

    def __getitem__(self, idx: int):
        return self._dataset[self._idxs[idx]]

    def get_eval_context(self, idx: int):
        return self._dataset.get_eval_context(self._idxs[idx])
