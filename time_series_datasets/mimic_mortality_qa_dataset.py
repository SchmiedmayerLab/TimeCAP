#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""MIMIC-IV chart-events mortality QA dataset."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MIMIC_DATA_DIR = "data/mimic-iv-clinical-database-demo-2.2"
MIMIC_MORTALITY_VALUES = ["survived", "died"]
MIMIC_TRUEFALSE_VALUES = ["True", "False"]

# An MHC-trained checkpoint reproduces label spaces it was fine-tuned on and
# answers True/False for anything else, whatever options the prompt lists — see
# blood_pressure_categories, where parse rate is 0.0000 on the canonical MHC
# prompt builder. "survived"/"died" is such an unseen space. Asking the same
# question with True/False keeps the model in distribution so the answer is
# parseable at all; "native" is kept to measure that gap rather than assume it.
MIMIC_LABEL_SPACES = ("native", "truefalse")

# Which outcome "True" names, under label_space="truefalse". Flipping this
# keeps the data, the channels and the Rules block identical and changes only
# what the yes/no question asks — so it isolates question form from signal. A
# model with a True bias scores near the majority rate under "survived" and
# near zero under "died" while having learned nothing, and comparing the two
# is what exposes that. It also swaps which class is positive, so F1 under
# "survived" is F1 on the majority class and is NOT comparable to F1 under
# "died" — read the prediction distribution, not the headline number.
MIMIC_POSITIVE_CLASSES = ("died", "survived")


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


# MHC training samples were uniformly 19 channels x 168 hourly steps, in a fixed
# channel order. This projects MIMIC onto that geometry to test whether the
# checkpoint's collapse to a constant answer is driven by input shape rather
# than by the absence of signal.
#
# Slot 5 is heart rate in both datasets — the one honest correspondence, matching
# on display name and unit alike. The other MIMIC channels are packed into the
# remaining *continuous* slots, which deliberately mislabels them (systolic blood
# pressure arrives announced as "iPhone step count"). That is the point: it holds
# geometry fixed while making the channel semantics wrong, so a gain over the
# native layout is attributable to shape, not meaning.
#
# Mean arterial pressure is dropped rather than mapped: only six continuous slots
# remain for seven channels, and MAP is ~(SBP + 2*DBP)/3, so it carries the least
# independent information once systolic and diastolic are both present.
MHC_LAYOUT_LABEL_MODES = ("mhc", "native")
MHC_N_CHANNELS = 19
MHC_N_STEPS = 168
MHC_HEART_RATE_SLOT = 5
MHC_SLOT_SOURCES: tuple[str | None, ...] = (
    "mimic:chart:systolic_bp",       #  0 iPhone step count
    "mimic:chart:diastolic_bp",      #  1 iPhone distance
    "mimic:chart:respiratory_rate",  #  2 flights climbed (iPhone)
    "mimic:chart:spo2",              #  3 Apple Watch step count
    "mimic:chart:temperature",       #  4 Apple Watch distance
    "mimic:chart:heart_rate",        #  5 heart rate            <- exact match
    "mimic:chart:glucose",           #  6 active energy
) + (None,) * 12                     #  7-18 binary slots, always absent


def _mhc_slot_labels() -> tuple[tuple[str, str | None, bool], ...]:
    """(display, unit, is_continuous) for each MHC slot, as training saw them."""
    from extractors import ChannelConfig
    from mhc.constants import CHANNEL_META, CHANNEL_NAMES, CONTINUOUS_CHANNELS

    config = ChannelConfig(
        names=CHANNEL_NAMES, meta=CHANNEL_META, continuous=CONTINUOUS_CHANNELS
    )
    return tuple(
        (
            config.display_name(channel),
            CHANNEL_META.get(channel, (None, None, None))[1],
            channel in CONTINUOUS_CHANNELS,
        )
        for channel in CHANNEL_NAMES
    )


def _select_channels(
    channels: "Sequence[str] | None",
) -> tuple[MIMICChannelSpec, ...]:
    """Restrict the chart channels to ``channels``, keeping declaration order.

    Accepts either the qualified name ("mimic:chart:heart_rate") or the bare
    suffix ("heart_rate"). Order is taken from MIMIC_CHART_CHANNELS rather than
    the argument, so a given set of channels always reaches the model in one
    order regardless of how it was spelled on the command line.
    """
    if channels is None:
        return MIMIC_CHART_CHANNELS

    by_name = {spec.name: spec for spec in MIMIC_CHART_CHANNELS}
    by_name.update(
        {spec.name.rsplit(":", 1)[-1]: spec for spec in MIMIC_CHART_CHANNELS}
    )
    requested = list(dict.fromkeys(channels))
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(
            f"Unknown MIMIC channels {unknown}. Available: "
            + ", ".join(spec.name.rsplit(":", 1)[-1] for spec in MIMIC_CHART_CHANNELS)
        )
    if not requested:
        raise ValueError("channels must name at least one chart channel")

    keep = {by_name[name].name for name in requested}
    return tuple(spec for spec in MIMIC_CHART_CHANNELS if spec.name in keep)


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
        label_space: str = "native",
        positive_class: str = "died",
        channels: Sequence[str] | None = None,
        mhc_layout: bool = False,
        mhc_layout_labels: str = "mhc",
    ) -> None:
        if label_space not in MIMIC_LABEL_SPACES:
            raise ValueError(
                f"label_space must be one of {MIMIC_LABEL_SPACES}, got {label_space!r}"
            )
        if positive_class not in MIMIC_POSITIVE_CLASSES:
            raise ValueError(
                f"positive_class must be one of {MIMIC_POSITIVE_CLASSES}, "
                f"got {positive_class!r}"
            )
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
        # Set before _build_samples: formatting reads these to pick the post_prompt.
        self.label_space = label_space
        self.positive_class = positive_class
        self.mhc_layout = bool(mhc_layout)
        # "mhc" announces each slot under the training channel's name, so the
        # geometry is right and the semantics are wrong. "native" keeps the same
        # slots but tells the truth about what occupies them. Running both is
        # what separates the effect of shape from the effect of meaning.
        if mhc_layout_labels not in MHC_LAYOUT_LABEL_MODES:
            raise ValueError(
                f"mhc_layout_labels must be one of {MHC_LAYOUT_LABEL_MODES}, "
                f"got {mhc_layout_labels!r}"
            )
        self.mhc_layout_labels = mhc_layout_labels
        if self.mhc_layout and channels is not None:
            raise ValueError(
                "channels and mhc_layout are mutually exclusive: the MHC layout "
                "fixes which channel occupies which of the 19 slots."
            )
        self.channel_specs = _select_channels(channels)
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
        answer = row["answer"]
        if self.label_space == "truefalse":
            # True names whichever outcome positive_class selects, matching the
            # question asked. get_eval_context is the only place ground truth is
            # produced, so a run can never mix polarity with its prompt.
            answer = "True" if answer == self.positive_class else "False"
        return recording, answer

    def _get_pre_prompt(self, _row: dict) -> str:
        return "You are an expert in ICU clinical time series analysis."

    def _get_post_prompt(self, row: dict) -> str:
        # Mirrors build_metabolic_post_prompt's shape exactly — question, blank
        # line, "Possible answers:", then the Rules block. That framing is what
        # the checkpoint was fine-tuned to answer under; phrasing that never
        # appeared in training (inline answers, no rules, a trailing "Answer:")
        # drops it back into its captioning prior instead of classifying.
        hours = row["observation_hours"]
        if self.label_space == "truefalse":
            if self.positive_class == "survived":
                # "discharged alive" rather than "discharged": every patient
                # leaves the hospital eventually, so discharge alone does not
                # name the outcome. This is hospital_expire_flag == 0.
                asked = "was this patient discharged alive from the hospital?"
            else:
                asked = "did this patient die in the hospital?"
            question = (
                f"Based on the first {hours} hours of ICU charted vital signs, "
                f"{asked}"
            )
            values = ", ".join(MIMIC_TRUEFALSE_VALUES)
        else:
            question = (
                f"Based on the first {hours} hours of ICU charted vital signs, "
                "did this patient survive or die in the hospital: survived or died?"
            )
            values = ", ".join(MIMIC_MORTALITY_VALUES)
        return (
            f"{question}\n\n"
            f"Possible answers: {values}\n\n"
            "Rules:\n"
            "- Base your answer on the sensor patterns above.\n"
            "- You MUST give a classification even if the signal is unclear — "
            "state limitations but still make your best guess.\n"
            "- Never respond with a question.\n"
            "- You MUST respond with ONLY <label>, exactly as written in the "
            "listed possible answers."
        )

    def _get_text_time_series(self, row: dict) -> list[tuple[str, np.ndarray]]:
        if self.mhc_layout:
            return self._get_mhc_layout_time_series(row)
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

    def _get_mhc_layout_time_series(
        self, row: dict
    ) -> list[tuple[str, np.ndarray]]:
        """Project the stay onto 19 channels x 168 steps in MHC slot order.

        Observed hours occupy the front of the window and the remainder is left
        absent, which normalization maps to 0.0 — the same value MHC's own
        unobserved hours take, so the padding is not a distinguishable token.
        """
        values = np.asarray(row["values"], dtype=np.float32)
        by_name = {name: idx for idx, name in enumerate(row["channel_names"])}
        labels = _mhc_slot_labels()
        n_observed = min(values.shape[1], MHC_N_STEPS)

        prompts: list[tuple[str, np.ndarray]] = []
        mimic_specs = {spec.name: spec for spec in MIMIC_CHART_CHANNELS}
        for slot in range(MHC_N_CHANNELS):
            display, unit, is_continuous = labels[slot]
            source = MHC_SLOT_SOURCES[slot]
            series = np.full(MHC_N_STEPS, np.nan, dtype=np.float32)
            if source is not None and source in by_name:
                series[:n_observed] = values[by_name[source]][:n_observed]
                if self.mhc_layout_labels == "native":
                    # Same slot, same position, honest name — the only thing that
                    # changes relative to the "mhc" mode is what the model is told.
                    display = mimic_specs[source].display_name
                    unit = mimic_specs[source].unit

            if is_continuous:
                unit_str = f" in {unit}" if unit else ""
                prompts.append(self._continuous_prompt(series, display, unit_str))
            else:
                # Binary slots have no MIMIC counterpart. All-zero is a state MHC
                # produces whenever the user logged no such activity, so these
                # read as "nothing happened", not as malformed input.
                prompts.append(self._binary_prompt(series, display))
        return prompts

    @staticmethod
    def _binary_prompt(channel: np.ndarray, display: str) -> tuple[str, np.ndarray]:
        channel = np.nan_to_num(np.asarray(channel, dtype=np.float32), nan=0.0)
        text = (
            f"The following is a per-minute binary mask for {display} "
            f"(1 = active, 0 = inactive or nonwear):"
        )
        return text, channel

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

    @property
    def model_channel_count(self) -> int:
        """Series the model receives, which the MHC layout fixes at 19."""
        return MHC_N_CHANNELS if self.mhc_layout else len(self.channel_specs)

    @property
    def model_n_steps(self) -> int:
        """Length of each series the model receives; 168 under the MHC layout."""
        return MHC_N_STEPS if self.mhc_layout else self.n_bins

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
            "channel_count": self.model_channel_count,
            "input_shape": input_shape,
            "model_input_shape": [self.model_channel_count, self.model_n_steps],
            "mhc_layout": self.mhc_layout,
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
            if len(item["time_series"]) != self.model_channel_count:
                raise ValueError(
                    f"MIMIC formatted sample {idx} has {len(item['time_series'])} "
                    f"time-series channels, expected {self.model_channel_count}"
                )
            for channel in item["time_series"]:
                if len(channel) != self.model_n_steps:
                    raise ValueError(
                        f"MIMIC formatted sample {idx} has channel length "
                        f"{len(channel)}, expected {self.model_n_steps}"
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
