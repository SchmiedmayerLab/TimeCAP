#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Export the measurements behind the clinical captions as tabular features.

Runs every clinical synthesizer over the daily MHC dataset (minute resolution) and the
weekly ones over 7-day windows rebuilt from the same days at hourly resolution, then writes

  <out_dir>/daily.parquet                    one row per (user, day)
  <out_dir>/weekly.parquet                   one row per (user, week)
  <out_dir>/clinical_user_features.parquet   one row per user (mean / median / std / p10 /
                                             p90 of every value, plus each signal's flag
                                             rate), in the user-keyed format that
                                             MHC-benchmark's FE-XGBoost pathway reads

Each signal contributes its numeric values (``ClinicalSynthesizer.features``) and a flag
that mirrors how the caption pipeline reads it: 1 when its caption is flagged, 0 when it
was measured and not flagged, NaN when it could not be evaluated. ``n_*`` coverage columns
are dropped by the MHC-benchmark loader, so they never reach the classifier.

Example::

    python scripts/export_clinical_features.py \\
        --dataset_dir $MHC_DATASET_DIR \\
        --split_file data/splits/sharable_users_seed42_2026.json \\
        --out_dir clinical_features --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical.bradycardia import NocturnalBradycardiaSynthesizer
from clinical.cadence import CadenceSynthesizer
from clinical.circadian_dipping import CircadianDippingSynthesizer
from clinical.constants import ACTIVITY_CHANNELS, WATCH_HR_CHANNEL
from clinical.data import (
    CHANNEL_NAMES,
    DAILY_CLINICAL_CONFIG,
    WEEKLY_CLINICAL_CONFIG,
    _transform_daily,
    wear_pct,
)
from clinical.exercise_volume import ExerciseVolumeSynthesizer
from clinical.gait_speed import GaitSpeedSynthesizer
from clinical.hr_inversion import HRInversionSynthesizer
from clinical.hr_recovery import HRRecoverySynthesizer
from clinical.hr_spread import HRSpreadSynthesizer
from clinical.nocturnal_activity import NocturnalActivitySynthesizer
from clinical.nocturnal_rhr import NocturnalRHRSynthesizer
from clinical.peak_hr_proxy import PeakHRProxySynthesizer
from clinical.resting_hr import RestingHRSynthesizer
from clinical.sedentary_burden import SedentaryBurdenSynthesizer
from clinical.sustained_bout import SustainedBoutSynthesizer
from timef.schema import Recording

# Keyed by the label each synthesizer puts on its captions.
DAILY_SIGNALS = {
    "resting_hr":            RestingHRSynthesizer(),
    "nocturnal_rhr":         NocturnalRHRSynthesizer(),
    "nocturnal_bradycardia": NocturnalBradycardiaSynthesizer(),
    "nocturnal_activity":    NocturnalActivitySynthesizer(),
    "hr_spread":             HRSpreadSynthesizer(),
    "sustained_bout":        SustainedBoutSynthesizer(),
    "cadence":               CadenceSynthesizer(),
    "circadian_dipping":     CircadianDippingSynthesizer(),
    "gait_speed":            GaitSpeedSynthesizer(),
    "sedentary_burden":      SedentaryBurdenSynthesizer(),
    "hr_recovery":           HRRecoverySynthesizer(),
    "hr_inversion":          HRInversionSynthesizer(),
}
WEEKLY_SIGNALS = {
    "exercise_volume": ExerciseVolumeSynthesizer(),
    "peak_hr_proxy":   PeakHRProxySynthesizer(),
}

# Signals that stay silent when the measurement is unremarkable, so a measured day without
# a caption counts as "not flagged". Sustained bout is the reverse: its silence is the
# finding, as in caption_grouper.ABSENCE_SIGNALS. Every other signal captions whenever it
# can evaluate, so silence there means "not evaluated".
SILENT_WHEN_NORMAL = {"nocturnal_bradycardia", "nocturnal_activity", "sedentary_burden", "hr_inversion"}
SILENT_WHEN_FLAGGED = {"sustained_bout"}

# Channels the weekly synthesizers read; everything else stays NaN in the weekly recordings.
WEEKLY_CHANNEL_IDXS = [CHANNEL_NAMES.index(WATCH_HR_CHANNEL)] + [
    CHANNEL_NAMES.index(ch) for ch in ACTIVITY_CHANNELS
]
DAILY_COLUMNS = ["values", "user_id", "date", "has_any_data", "minutes_nonzero_or_nan",
                 "channel_units", "total_nonwear_minutes"]

# Signals that survive hourly input (--hourly). HR recovery, sustained bout, gait speed and
# bradycardia need minute-level transients and are dropped; cadence, sedentary burden and
# nocturnal activity only resolve whole hours.
HOURLY_SIGNALS = ("resting_hr", "nocturnal_rhr", "circadian_dipping", "hr_spread", "hr_inversion",
                  "cadence", "sedentary_burden", "nocturnal_activity")
HR_IDX = CHANNEL_NAMES.index(WATCH_HR_CHANNEL)
# Hourly channels that are sums over the hour (steps, distance, flights); spread evenly.
HOURLY_SUM_IDXS = [0, 1, 2, 3, 4]


def signal_features(name: str, synth, rec: Recording, config) -> dict[str, float]:
    """``<signal>__<value>`` columns plus ``<signal>__flag`` for one recording."""
    values = synth.features(rec)
    annotations = synth.synthesize(rec, config)
    measured = any(np.isfinite(v) for v in values.values())
    if annotations:
        flag = float(any(str(a.label or "").endswith(":flag") for a in annotations))
    elif measured and name in SILENT_WHEN_FLAGGED:
        flag = 1.0
    elif measured and name in SILENT_WHEN_NORMAL:
        flag = 0.0
    else:
        flag = np.nan
    out = {f"{name}__{key}": value for key, value in values.items()}
    out[f"{name}__flag"] = flag
    return out


def _hourly(rec: Recording) -> np.ndarray:
    """(len(WEEKLY_CHANNEL_IDXS), 24) hourly means: HR in bpm, workouts as active fraction."""
    minutes = rec.values[WEEKLY_CHANNEL_IDXS].reshape(len(WEEKLY_CHANNEL_IDXS), 24, 60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN hours stay NaN
        return np.nanmean(minutes, axis=2).astype(np.float32)


def _process_chunk(dataset_dir: str, indices: list[int]) -> tuple[pd.DataFrame, list[tuple[str, str]], np.ndarray]:
    """Daily features and hourly weekly-channel aggregates for one chunk of dataset rows."""
    from datasets import load_from_disk

    ds = load_from_disk(dataset_dir).select(indices).select_columns(DAILY_COLUMNS).with_format("numpy")
    rows: list[dict] = []
    keys: list[tuple[str, str]] = []
    hourly: list[np.ndarray] = []
    for batch in ds.iter(batch_size=256):
        for i in range(len(batch["user_id"])):
            raw = {col: batch[col][i] for col in DAILY_COLUMNS}
            raw["user_id"], raw["date"] = str(raw["user_id"]), str(raw["date"])
            rec = _transform_daily(raw)
            row = {"user_id": rec.user_id, "date": rec.date, "wear_pct": rec.wear_pct}
            for name, synth in DAILY_SIGNALS.items():
                row.update(signal_features(name, synth, rec, DAILY_CLINICAL_CONFIG))
            rows.append(row)
            keys.append((rec.user_id, rec.date))
            hourly.append(_hourly(rec))
    stacked = np.stack(hourly) if hourly else np.empty((0, len(WEEKLY_CHANNEL_IDXS), 24), np.float32)
    return pd.DataFrame(rows), keys, stacked


def _upsampled_recording(user_id: str, date: str, hourly: np.ndarray, mask: np.ndarray) -> Recording:
    """Minute-level Recording from a daily_hourly_hf row: each hour repeated over its 60 minutes,
    hourly sums divided by 60, HR converted to bpm."""
    hourly = hourly.astype(np.float32).copy()
    hourly[mask.astype(bool)] = np.nan
    hourly[HR_IDX][hourly[HR_IDX] == 0] = np.nan
    hourly[HR_IDX] *= 60.0
    hourly[HOURLY_SUM_IDXS] /= 60.0
    values = np.repeat(hourly, 60, axis=1)
    return Recording(
        row_id=f"mhc:{user_id}:{date}",
        user_id=user_id,
        date=date,
        values=values,
        channel_names=tuple(CHANNEL_NAMES),
        display_names=tuple(CHANNEL_NAMES),
        units=tuple([""] * len(CHANNEL_NAMES)),
        has_any_data=np.isfinite(values).any(axis=1),
        channel_variance=np.zeros(len(CHANNEL_NAMES), dtype=np.float32),
    )


def _process_hourly_chunk(dataset_dir: str, indices: list[int]) -> tuple[pd.DataFrame, list[tuple[str, str]], np.ndarray]:
    """Like _process_chunk, for the hourly daily_hourly_hf dataset (already wear-filtered)."""
    from datasets import load_from_disk

    ds = load_from_disk(dataset_dir).select(indices).with_format("numpy")
    rows: list[dict] = []
    keys: list[tuple[str, str]] = []
    hourly: list[np.ndarray] = []
    for batch in ds.iter(batch_size=1024):
        for i in range(len(batch["user_id"])):
            rec = _upsampled_recording(str(batch["user_id"][i]), str(batch["date"][i]),
                                       batch["values"][i], batch["mask"][i])
            row = {"user_id": rec.user_id, "date": rec.date}
            for name in HOURLY_SIGNALS:
                row.update(signal_features(name, DAILY_SIGNALS[name], rec, DAILY_CLINICAL_CONFIG))
            rows.append(row)
            keys.append((rec.user_id, rec.date))
            hourly.append(_hourly(rec))
    stacked = np.stack(hourly) if hourly else np.empty((0, len(WEEKLY_CHANNEL_IDXS), 24), np.float32)
    return pd.DataFrame(rows), keys, stacked


def weekly_features(keys: list[tuple[str, str]], hourly: np.ndarray, min_days: int) -> pd.DataFrame:
    """Run the weekly synthesizers on 7-day windows anchored at each user's first day."""
    days = pd.DataFrame(keys, columns=["user_id", "date"])
    days["date"] = pd.to_datetime(days["date"])
    days["row"] = np.arange(len(days))
    first = days.groupby("user_id")["date"].transform("min")
    days["offset"] = (days["date"] - first).dt.days
    days["week"] = days["offset"] // 7

    rows: list[dict] = []
    for (user_id, _), week in days.groupby(["user_id", "week"], sort=False):
        if len(week) < min_days:
            continue
        week_start = week["date"].min() - pd.Timedelta(days=int(week["offset"].min() % 7))
        values = np.full((len(CHANNEL_NAMES), 168), np.nan, dtype=np.float32)
        for row_idx, offset in zip(week["row"], week["offset"] % 7):
            values[WEEKLY_CHANNEL_IDXS, offset * 24:(offset + 1) * 24] = hourly[row_idx]
        rec = Recording(
            row_id=f"mhc_weekly:{user_id}:{week_start.date()}",
            user_id=user_id,
            date=str(week_start.date()),
            values=values,
            channel_names=tuple(CHANNEL_NAMES),
            display_names=tuple(CHANNEL_NAMES),
            units=tuple([""] * len(CHANNEL_NAMES)),
            has_any_data=np.isfinite(values).any(axis=1),
            channel_variance=np.zeros(len(CHANNEL_NAMES), dtype=np.float32),
        )
        row = {"user_id": user_id, "week_start": rec.date, "n_days": len(week)}
        for name, synth in WEEKLY_SIGNALS.items():
            row.update(signal_features(name, synth, rec, WEEKLY_CLINICAL_CONFIG))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Per-user summary statistics of every value column and the rate of every flag."""
    value_cols = [c for c in df.columns if "__" in c and not c.endswith("__flag")]
    flag_cols = [c for c in df.columns if c.endswith("__flag")]
    grouped = df.groupby("user_id")
    return pd.concat([
        grouped[value_cols].mean().add_suffix("__mean"),
        grouped[value_cols].median().add_suffix("__median"),
        grouped[value_cols].std().add_suffix("__std"),
        grouped[value_cols].quantile(0.1).add_suffix("__p10"),
        grouped[value_cols].quantile(0.9).add_suffix("__p90"),
        grouped[flag_cols].mean().add_suffix("_rate"),
        grouped.size().rename(f"n_{unit}"),
    ], axis=1)


def _load_split_users(split_file: str) -> set[str]:
    data = json.loads(Path(split_file).read_text())
    if isinstance(data, list):
        return set(data)
    return set(data.get("train", [])) | set(data.get("validation", [])) | set(data.get("test", []))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export clinical-signal features for a tabular baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset_dir", default=os.environ.get("MHC_DATASET_DIR"),
                   help="Daily MHC HF dataset (minute resolution). Defaults to $MHC_DATASET_DIR.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split_file", default=None,
                   help="Restrict to the users of a split JSON (train/validation/test) or list.")
    p.add_argument("--min_wear_pct", type=float, default=50.0, help="Daily wear filter.")
    p.add_argument("--min_days_per_week", type=int, default=5,
                   help="Days required for a weekly window (MHC-benchmark uses 5 of 7).")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--chunk_size", type=int, default=2000, help="Rows per worker task.")
    p.add_argument("--max_rows", type=int, default=None, help="Cap rows after filtering (smoke tests).")
    p.add_argument("--hourly", action="store_true",
                   help="dataset_dir is MHC-benchmark's daily_hourly_hf (19x24, already wear-filtered); "
                        "only the signals that survive hourly resolution are exported.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.dataset_dir:
        sys.exit("Pass --dataset_dir or set MHC_DATASET_DIR.")
    from datasets import load_from_disk

    t0 = time.time()
    if args.hourly:
        meta = load_from_disk(args.dataset_dir).select_columns(["user_id"]).to_pandas()
        keep = np.ones(len(meta), dtype=bool)
    else:
        meta = load_from_disk(args.dataset_dir).select_columns(["user_id", "total_nonwear_minutes"]).to_pandas()
        keep = np.array([wear_pct(r) >= args.min_wear_pct
                         for r in meta[["total_nonwear_minutes"]].to_dict("records")])
    if args.split_file:
        keep &= meta["user_id"].isin(_load_split_users(args.split_file)).to_numpy()
    indices = np.flatnonzero(keep).tolist()[:args.max_rows]
    print(f"{len(indices)} of {len(meta)} days pass the filters "
          f"({meta.loc[indices, 'user_id'].nunique()} users)")

    chunks = [indices[i:i + args.chunk_size] for i in range(0, len(indices), args.chunk_size)]
    daily_parts: list[pd.DataFrame] = []
    keys: list[tuple[str, str]] = []
    hourly_parts: list[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        process = _process_hourly_chunk if args.hourly else _process_chunk
        futures = [pool.submit(process, args.dataset_dir, chunk) for chunk in chunks]
        for done, future in enumerate(as_completed(futures), 1):
            daily, chunk_keys, hourly = future.result()
            daily_parts.append(daily)
            keys.extend(chunk_keys)
            hourly_parts.append(hourly)
            print(f"  {done}/{len(chunks)} chunks ({time.time() - t0:.0f}s)", flush=True)

    daily = pd.concat(daily_parts, ignore_index=True)
    weekly = weekly_features(keys, np.concatenate(hourly_parts), args.min_days_per_week)
    users = summarize(daily, "days")
    if not weekly.empty:
        users = users.join(summarize(weekly, "weeks"), how="outer")
    users = users.reset_index()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_dir / "daily.parquet", index=False)
    weekly.to_parquet(out_dir / "weekly.parquet", index=False)
    users.to_parquet(out_dir / "clinical_user_features.parquet", index=False)
    n_features = sum(not c.startswith(("n_", "user_id")) for c in users.columns)
    print(f"Wrote {len(daily)} days, {len(weekly)} weeks, {len(users)} users x {n_features} "
          f"features to {out_dir} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
