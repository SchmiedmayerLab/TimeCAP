#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Activity recognition evaluation dataset over MHC recordings.

Each sample is a 60-minute time window from an MHC recording paired with a
binary ground-truth label ("yes" / "no") derived directly from the workout:*
binary channels recorded by Apple Watch/HealthKit — no manual annotation needed.

Positive samples  : windows centered around a confirmed workout event.
Negative samples  : windows with no workout label but meaningful sensor activity.
Class balance     : equal positive and negative samples per recording.
User-level split  : same SHA-1 bucketing as MHCCaptionQADataset to prevent leakage.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation.tasks.base import EvalSample
from mhc.constants import CONTINUOUS_CHANNELS, DATASET_DIR
from mhc.dataset import MHCDataset
from mhc.transformer import MHCTransformer
from timef.schema import Recording

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKOUT_CHANNEL_NAMES: list[str] = [
    "workout:HKWorkoutActivityTypeWalking",
    "workout:HKWorkoutActivityTypeCycling",
    "workout:HKWorkoutActivityTypeRunning",
    "workout:HKWorkoutActivityTypeOther",
    "workout:HKWorkoutActivityTypeMixedMetabolicCardioTraining",
    "workout:HKWorkoutActivityTypeTraditionalStrengthTraining",
    "workout:HKWorkoutActivityTypeElliptical",
    "workout:HKWorkoutActivityTypeHighIntensityIntervalTraining",
    "workout:HKWorkoutActivityTypeFunctionalStrengthTraining",
    "workout:HKWorkoutActivityTypeYoga",
]

WORKOUT_LABELS: dict[str, str] = {
    "workout:HKWorkoutActivityTypeWalking": "walking",
    "workout:HKWorkoutActivityTypeCycling": "cycling",
    "workout:HKWorkoutActivityTypeRunning": "running",
    "workout:HKWorkoutActivityTypeOther": "other workout",
    "workout:HKWorkoutActivityTypeMixedMetabolicCardioTraining": "mixed cardio training",
    "workout:HKWorkoutActivityTypeTraditionalStrengthTraining": "strength training",
    "workout:HKWorkoutActivityTypeElliptical": "elliptical",
    "workout:HKWorkoutActivityTypeHighIntensityIntervalTraining": "high-intensity interval training",
    "workout:HKWorkoutActivityTypeFunctionalStrengthTraining": "functional strength training",
    "workout:HKWorkoutActivityTypeYoga": "yoga",
}

HEART_RATE_CHANNEL = "hk_watch:HKQuantityTypeIdentifierHeartRate"

_TRAIN_FRAC = 0.8
_VAL_FRAC = 0.1


# ---------------------------------------------------------------------------
# Internal index entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SampleIndex:
    """Lightweight pointer to a sample; full Recording loaded lazily in __getitem__."""
    dataset_idx: int
    window_start: int
    window_end: int
    ground_truth: str        # "yes" | "no"
    workout_channel: str     # channel name or "none" for negatives


# ---------------------------------------------------------------------------
# User-level split (mirrors MHCCaptionQADataset)
# ---------------------------------------------------------------------------

def _assign_split(user_id: str) -> Literal["train", "validation", "test"]:
    digest = int(hashlib.sha1(user_id.encode()).hexdigest(), 16)
    bucket = (digest % 1000) / 1000.0
    if bucket < _TRAIN_FRAC:
        return "train"
    if bucket < _TRAIN_FRAC + _VAL_FRAC:
        return "validation"
    return "test"


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def _find_workout_windows(
    channel_data: np.ndarray,
    min_duration: int,
) -> list[tuple[int, int]]:
    """Return (start, end) pairs for contiguous blocks of value==1."""
    windows: list[tuple[int, int]] = []
    in_block = False
    start = 0
    for i, val in enumerate(channel_data):
        if val == 1 and not in_block:
            in_block = True
            start = i
        elif val != 1 and in_block:
            in_block = False
            if i - start >= min_duration:
                windows.append((start, i))
    if in_block and len(channel_data) - start >= min_duration:
        windows.append((start, len(channel_data)))
    return windows


def _center_window(
    event_start: int,
    event_end: int,
    window_size: int,
    max_len: int = 1440,
) -> tuple[int, int]:
    """Build a window_size window centered on [event_start, event_end]."""
    center = (event_start + event_end) // 2
    w_start = max(0, center - window_size // 2)
    w_end = min(max_len, w_start + window_size)
    w_start = max(0, w_end - window_size)
    return w_start, w_end


def _find_negative_windows(
    data: np.ndarray,
    channel_names: list[str],
    window_size: int,
    rng: np.random.RandomState,
    n: int,
    min_hr_coverage: float = 0.3,
) -> list[tuple[int, int]]:
    """Find windows with no workout label but meaningful HR activity.

    Args:
        min_hr_coverage: Fraction of window minutes that must have valid HR.
    """
    workout_idx = [i for i, name in enumerate(channel_names) if name in WORKOUT_CHANNEL_NAMES]
    hr_idx = next((i for i, name in enumerate(channel_names) if name == HEART_RATE_CHANNEL), None)

    # Minutes where no workout is active (NaN counts as "no workout")
    no_workout = np.ones(data.shape[1], dtype=bool)
    for idx in workout_idx:
        no_workout &= (data[idx] != 1)

    # Minutes with valid HR data
    if hr_idx is not None:
        hr = data[hr_idx]
        has_hr = np.isfinite(hr) & (hr > 0)
    else:
        has_hr = np.ones(data.shape[1], dtype=bool)

    # Candidate window starts with stride = window_size // 2
    stride = max(1, window_size // 2)
    valid_starts: list[int] = []
    for start in range(0, data.shape[1] - window_size + 1, stride):
        end = start + window_size
        window_no_workout = no_workout[start:end]
        window_has_hr = has_hr[start:end]
        if window_no_workout.all() and window_has_hr.mean() >= min_hr_coverage:
            valid_starts.append(start)

    if not valid_starts:
        return []

    chosen = rng.choice(valid_starts, size=min(n, len(valid_starts)), replace=False)
    return [(int(s), int(s + window_size)) for s in chosen]


# ---------------------------------------------------------------------------
# Windowed Recording helper
# ---------------------------------------------------------------------------

def _make_window_recording(
    recording: Recording,
    start: int,
    end: int,
) -> Recording:
    """Slice a Recording to [start, end) keeping only continuous channels.

    Workout channels are intentionally excluded — they are the labels, not inputs.
    """
    cont_idx = [
        i for i, name in enumerate(recording.channel_names)
        if name in CONTINUOUS_CHANNELS
    ]
    values_sliced = recording.values[np.ix_(cont_idx, np.arange(start, end))]
    n_ch = len(cont_idx)

    has_any = np.array(
        [bool(np.any(np.isfinite(values_sliced[j]) & (values_sliced[j] > 0)))
         for j in range(n_ch)],
        dtype=bool,
    )
    variance = np.array(
        [float(np.nanvar(values_sliced[j])) for j in range(n_ch)],
        dtype=np.float32,
    )
    nonzero_or_nan = np.array(
        [float(np.sum(~np.isnan(values_sliced[j]) & (values_sliced[j] > 0)))
         for j in range(n_ch)],
        dtype=np.float32,
    )

    return Recording(
        row_id=f"{recording.row_id}[{start}:{end}]",
        user_id=recording.user_id,
        date=recording.date,
        values=values_sliced,
        channel_names=tuple(recording.channel_names[i] for i in cont_idx),
        display_names=tuple(recording.display_names[i] for i in cont_idx),
        units=tuple(recording.units[i] for i in cont_idx),
        has_any_data=has_any,
        channel_variance=variance,
        minutes_nonzero_or_nan=nonzero_or_nan,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MHCActivityDataset(Dataset):
    """PyTorch Dataset of 60-minute activity windows from the MHC dataset.

    Ground truth comes directly from Apple Watch workout:* binary channels —
    no manual labelling required.

    Args:
        split: One of "train", "validation", "test".
        dataset_dir: Path to the HuggingFace MHC dataset on disk.
        window_size: Number of minutes per sample window.
        min_workout_duration: Minimum consecutive workout minutes to form a positive.
        max_recordings: Cap on recordings scanned (useful for development).
        seed: Random seed for negative sample selection.
    """

    def __init__(
        self,
        split: Literal["train", "validation", "test"],
        dataset_dir: str | Path = DATASET_DIR,
        window_size: int = 60,
        min_workout_duration: int = 5,
        max_recordings: int | None = None,
        seed: int = 42,
    ) -> None:
        self._raw = MHCDataset(dataset_dir=str(dataset_dir))
        self._transformer = MHCTransformer()
        self._window_size = window_size
        self._split = split
        self._rng = np.random.RandomState(seed)
        self._samples = self._build_sample_index(min_workout_duration, max_recordings)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EvalSample:
        s = self._samples[idx]
        raw_row = self._raw[s.dataset_idx]
        recording = self._transformer.transform_row(raw_row)
        windowed = _make_window_recording(recording, s.window_start, s.window_end)

        return EvalSample(
            sample_id=f"{recording.row_id}[{s.window_start}:{s.window_end}]:{s.workout_channel}",
            recording=windowed,
            ground_truth=s.ground_truth,
            metadata={
                "workout_channel": s.workout_channel,
                "workout_label": WORKOUT_LABELS.get(s.workout_channel, s.workout_channel),
                "window_start": s.window_start,
                "window_end": s.window_end,
                "window_size": s.window_end - s.window_start,
                "user_id": raw_row["user_id"],
                "date": raw_row["date"],
            },
        )

    # ------------------------------------------------------------------
    # Sample index construction
    # ------------------------------------------------------------------

    def _build_sample_index(
        self,
        min_workout_duration: int,
        max_recordings: int | None,
    ) -> list[_SampleIndex]:
        """Scan MHC recordings and build the flat list of sample pointers.

        Only raw numpy values and channel names are accessed here; full
        MHCTransformer conversion is deferred to __getitem__.
        """
        n_total = len(self._raw)
        indices = (
            self._rng.choice(n_total, size=min(max_recordings, n_total), replace=False).tolist()
            if max_recordings is not None
            else range(n_total)
        )

        samples: list[_SampleIndex] = []

        for dataset_idx in tqdm(indices, desc=f"Scanning MHC ({self._split})"):
            row = self._raw[dataset_idx]

            if _assign_split(row["user_id"]) != self._split:
                continue

            channel_names: list[str] = row["channel_names"]
            data: np.ndarray = row["data"]  # (19, 1440)

            pos: list[_SampleIndex] = []

            for ch_idx, ch_name in enumerate(channel_names):
                if ch_name not in WORKOUT_CHANNEL_NAMES:
                    continue
                for event_start, event_end in _find_workout_windows(
                    data[ch_idx], min_workout_duration
                ):
                    w_start, w_end = _center_window(
                        event_start, event_end, self._window_size
                    )
                    pos.append(_SampleIndex(
                        dataset_idx=dataset_idx,
                        window_start=w_start,
                        window_end=w_end,
                        ground_truth="yes",
                        workout_channel=ch_name,
                    ))

            if not pos:
                continue

            neg_windows = _find_negative_windows(
                data, channel_names, self._window_size, self._rng, n=len(pos)
            )
            neg = [
                _SampleIndex(
                    dataset_idx=dataset_idx,
                    window_start=w_start,
                    window_end=w_end,
                    ground_truth="no",
                    workout_channel="none",
                )
                for w_start, w_end in neg_windows
            ]

            samples.extend(pos + neg)

        print(
            f"MHCActivityDataset ({self._split}): {len(samples)} samples "
            f"({sum(1 for s in samples if s.ground_truth == 'yes')} positive, "
            f"{sum(1 for s in samples if s.ground_truth == 'no')} negative)"
        )
        return samples
