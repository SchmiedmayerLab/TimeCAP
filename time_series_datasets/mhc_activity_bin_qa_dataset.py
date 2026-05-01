#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Bin-labelled activity classification QA dataset.

Sequence-labelling formulation: each item is one MHC recording (full
``window_minutes``-minute span — daily by default, weekly stride-7 also works)
paired with a structured **bin-labelled** answer string. The model sees the
full recording as input (identical shape to captioning) and must emit one
activity label per consecutive ``bin_size_min``-minute bin in chronological
order. The closed answer set per bin is ``ACTIVITY_CLASSES`` (10 workout
types + ``"no workout"``).

Differences from the older sliding-window approach:

* **One sample per recording**, not many overlapping windows. Subclasses
  ``MHCBaseQADataset`` directly because the 1-row → 1-sample contract is
  honoured.
* **Full-context input** — the same ``_get_text_time_series_prompt_list``
  shape captioning already uses, just with workout channels removed (they
  are the labels).
* **Structured sequence output** — answer is a comma-separated bin-label
  list of length ``n_bins`` exactly.

NaN minutes in workout channels are treated as inactive (value 0), matching
MHC-benchmark's convention in ``workout_gap.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal

import numpy as np
import torch

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt

from mhc.constants import ACTIVITY_CHANNELS, CONTINUOUS_CHANNELS
from time_series_datasets.mhc_base_qa_dataset import MHCBaseQADataset
from time_series_datasets.mhc_label_lookup import (
    ACTIVITY_CLASSES,
    WORKOUT_LABELS,
    build_activity_bin_post_prompt,
)


DEFAULT_BIN_SIZE_MIN = 30
DEFAULT_MIN_ACTIVE = 5
NO_WORKOUT_LABEL = "no workout"


def _label_dominant_workout(
    data: np.ndarray,
    channel_names: list[str],
    w_start: int,
    w_end: int,
    min_active: int,
) -> str:
    """Return the workout class with the most active minutes in [w_start, w_end).

    Active = ``value == 1``. NaN minutes are treated as 0 (MHC convention).
    Returns ``NO_WORKOUT_LABEL`` if no workout channel reaches ``min_active``.
    """
    best_label = NO_WORKOUT_LABEL
    best_count = 0
    for ch_idx, ch_name in enumerate(channel_names):
        if ch_name not in ACTIVITY_CHANNELS:
            continue
        active = int(
            (np.nan_to_num(data[ch_idx, w_start:w_end], nan=0.0) == 1).sum()
        )
        if active >= min_active and active > best_count:
            best_count = active
            best_label = WORKOUT_LABELS[ch_name]
    return best_label


def _compute_bin_labels(
    values: np.ndarray,
    channel_names: list[str],
    bin_size_min: int,
    min_active: int,
) -> list[str]:
    """Bin the recording into consecutive ``bin_size_min``-minute slots and
    return the dominant-workout label for each. Trailing partial bin is dropped.
    """
    n_minutes = values.shape[1]
    n_bins = n_minutes // bin_size_min
    labels: list[str] = []
    for b in range(n_bins):
        start = b * bin_size_min
        end = start + bin_size_min
        labels.append(
            _label_dominant_workout(values, channel_names, start, end, min_active)
        )
    return labels


class MHCActivityBinQADataset(MHCBaseQADataset):
    """Full-recording in, ``n_bins`` activity labels out (one per bin).

    Args mirror ``MHCBaseQADataset`` plus:
        bin_size_min: minutes per bin (default 30).
        min_active: minimum active minutes within a bin to label it as a
            workout class instead of ``"no workout"`` (default 5).
    """

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        export_dir: str | Path,
        bin_size_min: int = DEFAULT_BIN_SIZE_MIN,
        min_active: int = DEFAULT_MIN_ACTIVE,
    ) -> None:
        self._bin_size_min = bin_size_min
        self._min_active = min_active
        super().__init__(split, EOS_TOKEN, export_dir=export_dir)
        # ``MHCBaseQADataset.__init__`` populates self._raw_samples; the bin
        # count is uniform across samples within the same export resolution,
        # so we cache n_bins from the first sample for the post_prompt.
        if self._raw_samples:
            n_minutes = np.asarray(
                self._raw_samples[0]["values"], dtype=np.float32
            ).shape[1]
            self._n_bins = n_minutes // bin_size_min
        else:
            self._n_bins = 0

    def _get_answer(self, row) -> str:
        values = np.asarray(row["values"], dtype=np.float32)
        labels = _compute_bin_labels(
            values, row["channel_names"], self._bin_size_min, self._min_active,
        )
        return ", ".join(labels)

    def _get_post_prompt(self, _row) -> str:
        return build_activity_bin_post_prompt(self._n_bins, self._bin_size_min)

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """Same shape as captioning — but workout channels are stripped because
        they are the labels (would make classification trivial)."""
        values = np.asarray(row["values"], dtype=np.float32)
        channel_names: list[str] = row["channel_names"]
        display_names: list[str] = row["display_names"]
        units: list[str | None] = row["units"]

        prompts: list[TextTimeSeriesPrompt] = []
        for idx, name in enumerate(channel_names):
            if name in ACTIVITY_CHANNELS:
                continue  # workout channels are labels, not inputs
            channel = torch.from_numpy(values[idx])
            display = display_names[idx]
            unit = units[idx]
            unit_str = f" in {unit}" if unit else ""
            if name in CONTINUOUS_CHANNELS:
                prompts.append(MHCBaseQADataset._continuous_prompt(channel, display, unit_str))
            else:
                prompts.append(MHCBaseQADataset._binary_prompt(channel, display))
        return prompts
