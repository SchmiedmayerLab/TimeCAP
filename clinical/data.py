#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Self-contained MHC data loader for the clinical audit.

Replicates the logic of MHCTransformer and WeeklyMHCTransformer without
importing the mhc/mhc_weekly packages (their __init__.py eagerly imports
MHCDataset -> torch, which is unavailable in the audit environment).

Reads MHC_DATASET_DIR / MHC_WEEKLY_DATASET_DIR env vars, the same as the
main pipeline.
"""
from __future__ import annotations

import os
from typing import Generator

import numpy as np

from clinical.constants import (
    ACTIVITY_CHANNELS,
    SLEEP_ASLEEP_CHANNEL,
    SLEEP_INBED_CHANNEL,
    WATCH_ACTIVE_ENERGY_CHANNEL,
    WATCH_DISTANCE_CHANNEL,
    WATCH_HR_CHANNEL,
    WATCH_STEP_CHANNEL,
)
from timef.schema import Recording

# ── Channel lists (mirrors mhc/constants.py CHANNEL_NAMES) ──────────────────

CHANNEL_NAMES: list[str] = [
    "hk_iphone:HKQuantityTypeIdentifierStepCount",
    "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning",
    "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed",
    WATCH_STEP_CHANNEL,
    WATCH_DISTANCE_CHANNEL,
    WATCH_HR_CHANNEL,
    WATCH_ACTIVE_ENERGY_CHANNEL,
    SLEEP_ASLEEP_CHANNEL,
    SLEEP_INBED_CHANNEL,
] + ACTIVITY_CHANNELS

HR_CHANNEL_IDX = CHANNEL_NAMES.index(WATCH_HR_CHANNEL)   # 5


def _compute_variance(data: np.ndarray) -> np.ndarray:
    out = np.zeros(data.shape[0], dtype=np.float32)
    for i, ch in enumerate(data):
        valid = ch[np.isfinite(ch)]
        if valid.size:
            out[i] = float(np.var(valid))
    return out


# ── Daily loader ─────────────────────────────────────────────────────────────

def iter_daily_recordings(
    dataset_dir: str | None = None,
    min_wear_pct: float = 0.0,
    split_users: set[str] | None = None,
    max_rows: int | None = None,
) -> Generator[Recording, None, None]:
    """Yield Recording objects from the daily MHC dataset (1440-min windows)."""
    from datasets import load_from_disk  # noqa: PLC0415
    path = dataset_dir or os.environ.get("MHC_DATASET_DIR", "")
    if not path:
        raise RuntimeError("Set MHC_DATASET_DIR env var or pass dataset_dir.")
    ds = load_from_disk(path)

    seen = 0
    for row in ds:
        if max_rows is not None and seen >= max_rows:
            break
        if split_users is not None and row["user_id"] not in split_users:
            continue
        wear_pct = row.get("wear_pct", 100.0) or 0.0
        if wear_pct < min_wear_pct:
            continue
        yield _transform_daily(row)
        seen += 1


def _transform_daily(row: dict) -> Recording:
    user_id = row["user_id"]
    date = row["date"]
    row_id = f"mhc:{user_id}:{date}"
    data = np.asarray(row["values"], dtype=np.float32).copy()  # (19, 1440)

    # Zero-only channels → NaN (same as MHCTransformer)
    data[HR_CHANNEL_IDX][data[HR_CHANNEL_IDX] == 0] = np.nan
    for ch_idx in (0, 1, 3, 4, 6):
        if np.all(data[ch_idx] == 0):
            data[ch_idx] = np.nan

    # HR: beats/sec → bpm
    data[HR_CHANNEL_IDX] *= 60.0

    raw_has_any = row.get("has_any_data")
    if raw_has_any is not None:
        has_any_data = np.asarray(raw_has_any, dtype=bool)
        if has_any_data.shape != (len(CHANNEL_NAMES),):
            has_any_data = np.array([np.any(np.isfinite(ch)) for ch in data], dtype=bool)
    else:
        has_any_data = np.array([np.any(np.isfinite(ch)) for ch in data], dtype=bool)

    raw_nonzero = row.get("minutes_nonzero_or_nan")
    if raw_nonzero is not None:
        minutes_nonzero = np.asarray(raw_nonzero, dtype=np.float32)
        if minutes_nonzero.shape != (len(CHANNEL_NAMES),):
            minutes_nonzero = None
    else:
        minutes_nonzero = None

    channel_units = row.get("channel_units", [""] * len(CHANNEL_NAMES))

    return Recording(
        row_id=row_id,
        user_id=user_id,
        date=date,
        values=np.ascontiguousarray(data),
        channel_names=tuple(CHANNEL_NAMES),
        display_names=tuple(CHANNEL_NAMES),
        units=tuple(channel_units),
        has_any_data=has_any_data,
        minutes_nonzero_or_nan=minutes_nonzero,
        channel_variance=_compute_variance(data),
        total_nonwear_minutes=row.get("total_nonwear_minutes"),
        wear_pct=row.get("wear_pct"),
    )


# ── Weekly loader ─────────────────────────────────────────────────────────────

def iter_weekly_recordings(
    dataset_dir: str | None = None,
    min_valid_hours: int = 0,
    split_users: set[str] | None = None,
    max_rows: int | None = None,
) -> Generator[Recording, None, None]:
    """Yield Recording objects from the weekly MHC dataset (168-hour windows)."""
    from datasets import load_from_disk  # noqa: PLC0415
    path = dataset_dir or os.environ.get("MHC_WEEKLY_DATASET_DIR", "")
    if not path:
        raise RuntimeError("Set MHC_WEEKLY_DATASET_DIR env var or pass dataset_dir.")
    ds = load_from_disk(path)

    seen = 0
    for row in ds:
        if max_rows is not None and seen >= max_rows:
            break
        if split_users is not None and row["user_id"] not in split_users:
            continue
        if row.get("n_valid_hours", 168) < min_valid_hours:
            continue
        yield _transform_weekly(row)
        seen += 1


def _transform_weekly(row: dict) -> Recording:
    user_id = row["user_id"]
    week_start = str(row["week_start"])
    row_id = f"mhc_weekly:{user_id}:{week_start}"

    # Dataset stores (168, 19); Recording needs (channels=19, time=168)
    data = np.asarray(row["values"], dtype=np.float32).T.copy()  # (19, 168)

    # mask=1 means missing → NaN
    raw_mask = row.get("mask")
    if raw_mask is not None:
        missing = np.asarray(raw_mask, dtype=bool).T  # (19, 168)
        data[missing] = np.nan

    # HR: beats/sec → bpm
    data[HR_CHANNEL_IDX] *= 60.0
    # Active energy: mean cal/min per hour → kcal/hr
    energy_idx = CHANNEL_NAMES.index(WATCH_ACTIVE_ENERGY_CHANNEL)
    data[energy_idx] *= 60.0 / 1000.0

    has_any_data = np.array([bool(np.any(np.isfinite(ch) & (ch != 0))) for ch in data], dtype=bool)

    return Recording(
        row_id=row_id,
        user_id=user_id,
        date=week_start,
        values=np.ascontiguousarray(data),
        channel_names=tuple(CHANNEL_NAMES),
        display_names=tuple(CHANNEL_NAMES),
        units=tuple([""] * len(CHANNEL_NAMES)),
        has_any_data=has_any_data,
        channel_variance=_compute_variance(data),
    )


# ── Minimal ChannelConfig stub for clinical synthesizers ────────────────────

class _ClinicalChannelConfig:
    """Minimal duck-type of ChannelConfig needed by ClinicalSynthesizer.synthesize().

    Clinical synthesizers only look at row.channel_names / row.values directly
    and don't use config fields, so this stub satisfies the signature.
    """
    time_unit: str = "minutes"
    names: list[str] = CHANNEL_NAMES


DAILY_CLINICAL_CONFIG = _ClinicalChannelConfig()
WEEKLY_CLINICAL_CONFIG = _ClinicalChannelConfig()
WEEKLY_CLINICAL_CONFIG.time_unit = "hours"  # type: ignore[assignment]
