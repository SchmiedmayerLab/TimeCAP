#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""MIMIC-IV chart-events mortality QA dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MIMIC_DATA_DIR = "data/mimic-iv-clinical-database-demo-2.2"
MIMIC_MORTALITY_VALUES = ["survived", "died"]


@dataclass(frozen=True)
class MIMICChannelSpec:
    name: str
    display_name: str
    unit: str | None
    itemids: tuple[int, ...]
    fahrenheit_itemids: tuple[int, ...] = ()


MIMIC_CHART_CHANNELS = (
    MIMICChannelSpec(
        name="mimic:chart:heart_rate",
        display_name="heart rate",
        unit="bpm",
        itemids=(220045,),
    ),
    MIMICChannelSpec(
        name="mimic:chart:systolic_bp",
        display_name="systolic blood pressure",
        unit="mmHg",
        itemids=(220050, 220179),
    ),
    MIMICChannelSpec(
        name="mimic:chart:diastolic_bp",
        display_name="diastolic blood pressure",
        unit="mmHg",
        itemids=(220051, 220180),
    ),
    MIMICChannelSpec(
        name="mimic:chart:mean_bp",
        display_name="mean arterial pressure",
        unit="mmHg",
        itemids=(220052, 220181),
    ),
    MIMICChannelSpec(
        name="mimic:chart:respiratory_rate",
        display_name="respiratory rate",
        unit="insp/min",
        itemids=(220210,),
    ),
    MIMICChannelSpec(
        name="mimic:chart:spo2",
        display_name="oxygen saturation",
        unit="%",
        itemids=(220277,),
    ),
    MIMICChannelSpec(
        name="mimic:chart:temperature",
        display_name="temperature",
        unit="degC",
        itemids=(223761, 223762, 226329),
        fahrenheit_itemids=(223761,),
    ),
    MIMICChannelSpec(
        name="mimic:chart:glucose",
        display_name="glucose",
        unit="mg/dL",
        itemids=(220621, 225664, 226537),
    ),
)


class MIMICMortalityQADataset:
    """Format MIMIC-IV chart events as in-hospital mortality QA samples.

    The adapter expects the minimum clinical-table set:
    admissions, icustays, d_items, and chartevents. It supports both the demo
    flat layout and the official MIMIC-IV ``hosp/`` and ``icu/`` subfolders.
    """

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_MIMIC_DATA_DIR,
        observation_hours: int = 24,
        bin_hours: int = 1,
        min_observed_channels: int = 1,
        require_full_observation_window: bool = True,
        chunk_rows: int = 1_000_000,
        eos_token: str = "",
    ) -> None:
        if observation_hours <= 0:
            raise ValueError("observation_hours must be positive")
        if bin_hours <= 0:
            raise ValueError("bin_hours must be positive")
        if min_observed_channels <= 0:
            raise ValueError("min_observed_channels must be positive")
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")

        self.data_dir = Path(data_dir)
        self.observation_hours = int(observation_hours)
        self.bin_hours = int(bin_hours)
        self.n_bins = int(np.ceil(self.observation_hours / self.bin_hours))
        self.min_observed_channels = int(min_observed_channels)
        self.require_full_observation_window = bool(require_full_observation_window)
        self.chunk_rows = int(chunk_rows)
        self.eos_token = eos_token
        self.channel_specs = MIMIC_CHART_CHANNELS
        self._raw_samples: list[dict] = []
        self.items: list[dict] = []

        paths = _resolve_mimic_paths(self.data_dir)
        # d_items is resolved to validate the minimum MIMIC-IV file bundle. The
        # first adapter version uses fixed itemids so it can stream chartevents.
        _ = paths["d_items"]
        stays = self._load_stays(paths["icustays"], paths["admissions"])
        measurements = self._load_chart_measurements(paths["chartevents"], stays)
        self._build_samples(stays, measurements)
        self.assert_no_model_input_leakage()

    def _load_stays(self, icustays_path: Path, admissions_path: Path) -> pd.DataFrame:
        admissions = pd.read_csv(
            admissions_path,
            usecols=["subject_id", "hadm_id", "hospital_expire_flag"],
        )
        stays = pd.read_csv(
            icustays_path,
            usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        )
        stays["intime"] = pd.to_datetime(stays["intime"], errors="coerce")
        stays["outtime"] = pd.to_datetime(stays["outtime"], errors="coerce")
        stays = stays.dropna(subset=["intime", "stay_id", "hadm_id"])
        stays = stays.merge(
            admissions[["hadm_id", "hospital_expire_flag"]],
            on="hadm_id",
            how="inner",
            validate="many_to_one",
        )
        if self.require_full_observation_window:
            window_end = stays["intime"] + pd.to_timedelta(
                self.observation_hours,
                unit="h",
            )
            stays = stays[stays["outtime"] >= window_end].copy()
        stays["answer"] = np.where(
            stays["hospital_expire_flag"].astype(int) == 1,
            "died",
            "survived",
        )
        return stays

    def _load_chart_measurements(
        self,
        chartevents_path: Path,
        stays: pd.DataFrame,
    ) -> pd.DataFrame:
        item_to_channel = {
            itemid: spec.name
            for spec in self.channel_specs
            for itemid in spec.itemids
        }
        target_itemids = set(item_to_channel)
        stay_lookup = stays[["stay_id", "intime"]].copy()
        stay_lookup["stay_id"] = stay_lookup["stay_id"].astype("int64")
        chunks: list[pd.DataFrame] = []

        usecols = ["stay_id", "charttime", "itemid", "valuenum", "valueuom"]
        for chunk in pd.read_csv(
            chartevents_path,
            usecols=usecols,
            chunksize=self.chunk_rows,
            low_memory=False,
        ):
            chunk = chunk[
                chunk["itemid"].isin(target_itemids)
                & chunk["valuenum"].notna()
                & chunk["stay_id"].notna()
            ].copy()
            if chunk.empty:
                continue

            chunk["stay_id"] = chunk["stay_id"].astype("int64")
            chunk = chunk.merge(stay_lookup, on="stay_id", how="inner")
            if chunk.empty:
                continue

            chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
            chunk = chunk.dropna(subset=["charttime"])
            elapsed_hours = (
                chunk["charttime"] - chunk["intime"]
            ).dt.total_seconds() / 3600.0
            chunk = chunk[
                (elapsed_hours >= 0)
                & (elapsed_hours < self.observation_hours)
            ].copy()
            if chunk.empty:
                continue

            chunk["hour_idx"] = np.floor(
                elapsed_hours.loc[chunk.index] / self.bin_hours
            ).astype("int64")
            chunk["channel_name"] = chunk["itemid"].map(item_to_channel)
            chunk["valuenum"] = _normalize_values(chunk)
            chunks.append(
                chunk[["stay_id", "channel_name", "hour_idx", "valuenum"]]
            )

        if not chunks:
            return pd.DataFrame(
                columns=["stay_id", "channel_name", "hour_idx", "valuenum"]
            )

        measurements = pd.concat(chunks, ignore_index=True)
        return (
            measurements.groupby(
                ["stay_id", "channel_name", "hour_idx"],
                as_index=False,
            )["valuenum"]
            .median()
        )

    def _build_samples(self, stays: pd.DataFrame, measurements: pd.DataFrame) -> None:
        channel_names = [spec.name for spec in self.channel_specs]
        display_names = [spec.display_name for spec in self.channel_specs]
        units = [spec.unit for spec in self.channel_specs]
        channel_idx = {name: idx for idx, name in enumerate(channel_names)}

        by_stay = {
            int(stay_id): group
            for stay_id, group in measurements.groupby("stay_id")
        }
        for row in stays.sort_values("stay_id").itertuples(index=False):
            stay_id = int(row.stay_id)
            group = by_stay.get(stay_id)
            if group is None or group.empty:
                continue

            values = np.full(
                (len(channel_names), self.n_bins),
                np.nan,
                dtype=np.float32,
            )
            for measurement in group.itertuples(index=False):
                idx = channel_idx.get(measurement.channel_name)
                hour_idx = int(measurement.hour_idx)
                if idx is not None and 0 <= hour_idx < self.n_bins:
                    values[idx, hour_idx] = float(measurement.valuenum)

            observed_channels = int(np.count_nonzero(np.any(np.isfinite(values), axis=1)))
            if observed_channels < self.min_observed_channels:
                continue

            sample = {
                "subject_id": str(row.subject_id),
                "hadm_id": str(row.hadm_id),
                "stay_id": str(stay_id),
                "date": row.intime.isoformat(),
                "observation_hours": self.observation_hours,
                "bin_hours": self.bin_hours,
                "observed_channels": observed_channels,
                "values": values,
                "channel_names": channel_names,
                "display_names": display_names,
                "units": units,
                "answer": row.answer,
            }
            self._raw_samples.append(sample)
            self.items.append(self._format_sample(sample))

    def _format_sample(self, row: dict) -> dict:
        time_series = self._get_text_time_series(row)
        return {
            "pre_prompt": self._get_pre_prompt(row),
            "time_series_text": [text for text, _tensor in time_series],
            "time_series": [tensor for _text, tensor in time_series],
            "post_prompt": self._get_post_prompt(row),
        }

    def get_eval_context(self, idx: int) -> tuple["Recording", str]:
        from timef.schema import Recording

        row = self._raw_samples[idx]
        values = np.asarray(row["values"], dtype=np.float32)
        recording = Recording(
            row_id=f"mimic:{row['stay_id']}",
            user_id=row["subject_id"],
            date=row["date"],
            values=values,
            channel_names=tuple(row["channel_names"]),
            display_names=tuple(row["display_names"]),
            units=tuple(row["units"]),
            has_any_data=np.any(~np.isnan(values), axis=1),
            channel_variance=np.nan_to_num(np.nanvar(values, axis=1), nan=0.0),
        )
        return recording, row["answer"]

    def _get_pre_prompt(self, _row: dict) -> str:
        return "You are an expert in ICU clinical time series analysis."

    def _get_post_prompt(self, row: dict) -> str:
        values = ", ".join(MIMIC_MORTALITY_VALUES)
        return (
            "Based on the first "
            f"{row['observation_hours']} hours of ICU charted vital signs, "
            "predict in-hospital mortality. Possible answers: "
            f"{values}\n\n"
            "Answer with ONLY one label from the possible answers.\n"
            "Answer:"
        )

    def _get_text_time_series(self, row: dict) -> list[tuple[str, np.ndarray]]:
        values = np.asarray(row["values"], dtype=np.float32)
        prompts: list[tuple[str, np.ndarray]] = []
        for idx, name in enumerate(row["channel_names"]):
            unit = row["units"][idx]
            unit_str = f" in {unit}" if unit else ""
            prompts.append(
                self._continuous_prompt(
                    values[idx],
                    row["display_names"][idx],
                    unit_str,
                )
            )
        return prompts

    @staticmethod
    def _continuous_prompt(
        channel: np.ndarray,
        display: str,
        unit_str: str,
    ) -> tuple[str, np.ndarray]:
        channel = np.asarray(channel, dtype=np.float32)
        finite_mask = np.isfinite(channel)
        if np.any(finite_mask):
            finite_vals = channel[finite_mask]
            mean = float(np.mean(finite_vals))
            std = max(float(np.std(finite_vals)), 1e-6)
        else:
            mean, std = 0.0, 1.0

        normalized = (channel - mean) / std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        text = (
            f"The following is the {display}{unit_str}, "
            f"it has mean {mean:.4f} and std {std:.4f}:"
        )
        return text, normalized.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]

    def label_counts(self) -> dict[str, int]:
        return {
            label: sum(1 for row in self._raw_samples if row["answer"] == label)
            for label in MIMIC_MORTALITY_VALUES
        }

    def cohort_summary(self) -> dict[str, object]:
        input_shape = None
        if self._raw_samples:
            values = self._raw_samples[0]["values"]
            input_shape = [int(values.shape[0]), int(values.shape[1])]
        return {
            "samples": len(self),
            "label_counts": self.label_counts(),
            "observation_hours": self.observation_hours,
            "bin_hours": self.bin_hours,
            "channel_count": len(self.channel_specs),
            "input_shape": input_shape,
            "min_observed_channels": self.min_observed_channels,
            "require_full_observation_window": self.require_full_observation_window,
        }

    def assert_no_model_input_leakage(self) -> None:
        allowed_keys = {"pre_prompt", "time_series_text", "time_series", "post_prompt"}
        forbidden = {
            "answer",
            "hospital_expire_flag",
            "deathtime",
            "dischtime",
            "outtime",
            "los",
            "discharge_location",
        }
        for idx, item in enumerate(self.items):
            keys = set(item)
            if keys != allowed_keys:
                raise ValueError(
                    f"MIMIC formatted sample {idx} has unexpected keys: {sorted(keys)}"
                )
            leaked = keys & forbidden
            if leaked:
                raise ValueError(
                    f"MIMIC formatted sample {idx} leaks outcome fields: {sorted(leaked)}"
                )
            if len(item["time_series"]) != len(self.channel_specs):
                raise ValueError(
                    f"MIMIC formatted sample {idx} has {len(item['time_series'])} "
                    f"time-series channels, expected {len(self.channel_specs)}"
                )
            for channel in item["time_series"]:
                if len(channel) != self.n_bins:
                    raise ValueError(
                        f"MIMIC formatted sample {idx} has channel length "
                        f"{len(channel)}, expected {self.n_bins}"
                    )


def _resolve_mimic_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "admissions": _first_existing(
            data_dir,
            (
                "admissions.csv.gz",
                "admissions.csv",
                "hosp/admissions.csv.gz",
                "hosp/admissions.csv",
            ),
        ),
        "icustays": _first_existing(
            data_dir,
            (
                "icustays.csv.gz",
                "icustays.csv",
                "icu/icustays.csv.gz",
                "icu/icustays.csv",
            ),
        ),
        "d_items": _first_existing(
            data_dir,
            (
                "d_items.csv.gz",
                "d_items.csv",
                "icu/d_items.csv.gz",
                "icu/d_items.csv",
            ),
        ),
        "chartevents": _first_existing(
            data_dir,
            (
                "chartevents.csv.gz",
                "chartevents.csv",
                "icu/chartevents.csv.gz",
                "icu/chartevents.csv",
            ),
        ),
    }


def _first_existing(data_dir: Path, candidates: tuple[str, ...]) -> Path:
    for candidate in candidates:
        path = data_dir / candidate
        if path.exists():
            return path
    expected = ", ".join(candidates)
    raise FileNotFoundError(f"Missing MIMIC-IV file under {data_dir}: {expected}")


def _normalize_values(chunk: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(chunk["valuenum"], errors="coerce").astype(float)
    fahrenheit_mask = (
        chunk["itemid"].isin({223761})
        | chunk["valueuom"].fillna("").str.contains("F", case=False, regex=False)
    )
    values.loc[fahrenheit_mask] = (values.loc[fahrenheit_mask] - 32.0) * (5.0 / 9.0)
    return values
