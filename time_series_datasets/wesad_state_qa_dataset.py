#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""WESAD state-classification QA dataset for cross-dataset evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import load_from_disk


WESAD_STATE_LABELS = {
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}
WESAD_STATE_VALUES = ["baseline", "stress", "amusement", "meditation"]
WESAD_STRESS_VALUES = ["True", "False"]

# One-vs-rest phrasing per state. Each is a yes/no question about a single
# state, so the model answers in the True/False space it was trained on rather
# than the 4-way space it has never seen.
WESAD_BINARY_QUESTIONS = {
    "stress": "Is this person stressed?",
    "baseline": "Is this person in a neutral, resting baseline state?",
    "amusement": "Is this person amused?",
    "meditation": "Is this person meditating?",
}
DEFAULT_WESAD_TIMEF_DATASET_DIR = "data/wesad_timef"
WESAD_LABEL_CHANNEL_PREFIX = "wesad:label:"
WESAD_DISPLAY_NAMES = {
    "wesad:chest:acc_x": "chest acc x",
    "wesad:chest:acc_y": "chest acc y",
    "wesad:chest:acc_z": "chest acc z",
    "wesad:chest:acc_mag": "chest acc magnitude",
    "wesad:chest:emg": "chest emg",
    "wesad:chest:eda": "chest eda",
    "wesad:chest:temp": "chest temperature",
    "wesad:chest:resp": "chest respiration magnitude",
    "wesad:wrist:acc_x": "wrist acc x",
    "wesad:wrist:acc_y": "wrist acc y",
    "wesad:wrist:acc_z": "wrist acc z",
    "wesad:wrist:acc_mag": "wrist acc magnitude",
    "wesad:wrist:bvp": "wrist bvp magnitude",
    "wesad:wrist:eda": "wrist eda",
    "wesad:wrist:temp": "wrist temperature",
    "wesad:e4:hr": "wrist heart rate",
}
_LABEL_SAMPLE_RATE_HZ = 700


class WESADStateQADataset:
    """Format WESAD labeled windows as model-facing QA samples.

    The stored WESAD dataset carries raw 700 Hz label ids plus per-second
    sensor channels. This adapter cuts fixed-width sensor windows, assigns each
    window its majority state, and removes WESAD label channels from inputs so
    evaluation cannot leak the answer.
    """

    def __init__(
        self,
        dataset_dir: str | Path = DEFAULT_WESAD_TIMEF_DATASET_DIR,
        window_seconds: int = 120,
        min_label_purity: float = 0.8,
        eos_token: str = "",
        binary_label: str | None = None,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if not 0.0 < min_label_purity <= 1.0:
            raise ValueError("min_label_purity must be in (0, 1]")
        if binary_label is not None and binary_label not in WESAD_BINARY_QUESTIONS:
            raise ValueError(
                f"binary_label must be one of {sorted(WESAD_BINARY_QUESTIONS)}, "
                f"got {binary_label!r}"
            )

        self.dataset_dir = Path(dataset_dir)
        self.window_seconds = int(window_seconds)
        self.min_label_purity = float(min_label_purity)
        self.eos_token = eos_token
        # Set before the load loop below: _add_row_windows formats each sample
        # as it is read, and formatting reads this to pick the post_prompt.
        self.binary_label = binary_label
        self._raw_samples: list[dict] = []
        self.items: list[dict] = []

        ds = load_from_disk(str(self.dataset_dir))
        for row in ds:
            self._add_row_windows(row)

    def _add_row_windows(self, row: dict) -> None:
        values = np.asarray(row["values"], dtype=np.float32)
        channel_names = list(row["channel_names"])
        channel_units = list(row["channel_units"])
        keep_idxs = [
            idx for idx, name in enumerate(channel_names)
            if not name.startswith(WESAD_LABEL_CHANNEL_PREFIX)
        ]
        if not keep_idxs:
            raise ValueError(f"No non-label WESAD channels found for {row['user_id']}")

        input_values = values[keep_idxs]
        input_names = [channel_names[idx] for idx in keep_idxs]
        input_units = [channel_units[idx] for idx in keep_idxs]
        display_names = [
            WESAD_DISPLAY_NAMES.get(name, name)
            for name in input_names
        ]

        per_second_labels = _majority_labels_per_second(row["label_values"])
        n_seconds = min(input_values.shape[1], len(per_second_labels))
        n_windows = n_seconds // self.window_seconds

        for window_idx in range(n_windows):
            start = window_idx * self.window_seconds
            end = start + self.window_seconds
            label, purity = _window_majority_label(
                per_second_labels[start:end],
                allowed_labels=WESAD_STATE_LABELS,
            )
            if label is None or purity < self.min_label_purity:
                continue

            sample = {
                "user_id": row["user_id"],
                "date": row["date"],
                "start_second": start,
                "end_second": end,
                "label_purity": purity,
                "values": input_values[:, start:end].copy(),
                "channel_names": input_names,
                "display_names": display_names,
                "units": input_units,
                "answer": WESAD_STATE_LABELS[label],
            }
            self._raw_samples.append(sample)
            self.items.append(self._format_sample(sample))

    def _format_sample(self, row: dict) -> dict:
        time_series = self._get_text_time_series(row)
        return {
            "pre_prompt": self._get_pre_prompt(row),
            "time_series_text": [
                text for text, _tensor in time_series
            ],
            "time_series": [
                tensor for _text, tensor in time_series
            ],
            "post_prompt": self._get_post_prompt(row),
        }

    def get_eval_context(self, idx: int) -> tuple[Recording, str]:
        from timef.schema import Recording

        row = self._raw_samples[idx]
        values = np.asarray(row["values"], dtype=np.float32)
        recording = Recording(
            row_id=(
                f"wesad:{row['user_id']}:{row['start_second']}"
                f"-{row['end_second']}"
            ),
            user_id=row["user_id"],
            date=row["date"],
            values=values,
            channel_names=tuple(row["channel_names"]),
            display_names=tuple(row["display_names"]),
            units=tuple(row["units"]),
            has_any_data=np.any(~np.isnan(values), axis=1),
            channel_variance=np.nan_to_num(np.nanvar(values, axis=1), nan=0.0),
        )
        answer = row["answer"]
        if self.binary_label is not None:
            # One-vs-rest: the asked-about state is positive, all three others
            # are negative. get_eval_context is the only place ground truth is
            # produced, so a run can never mix framings.
            answer = "True" if answer == self.binary_label else "False"
        return recording, answer

    def _get_pre_prompt(self, _row: dict) -> str:
        # Sentence one is byte-identical to MHC's pre_prompt. Sentence two states
        # the sampling rate, which no prompt in this codebase has ever carried:
        # MHC's per-channel text names a unit but never a time base, so the model
        # only ever learned "hour" from its caption *targets*
        # (templates_hourly.json). Handed a 120-step WESAD window it narrates in
        # hours regardless, so the granularity has to be said out loud somewhere.
        return (
            "You are an expert in wearable sensor time series analysis. "
            f"The following signals are sampled once per second and cover a "
            f"{self.window_seconds}-second window."
        )

    def _get_post_prompt(self, _row: dict) -> str:
        # Mirrors build_metabolic_post_prompt's shape exactly — question, blank
        # line, "Possible answers:", then the Rules block. That framing is what
        # the checkpoint was fine-tuned to answer under; the previous phrasing
        # (inline answers, no rules, trailing "Answer:") never appeared in
        # training, and the model fell back to captioning instead of classifying.
        if self.binary_label is not None:
            question = WESAD_BINARY_QUESTIONS[self.binary_label]
            values = ", ".join(WESAD_STRESS_VALUES)
        else:
            question = "What state is this person in?"
            values = ", ".join(WESAD_STATE_VALUES)
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
        values = np.asarray(row["values"], dtype=np.float32)
        prompts: list[tuple[str, np.ndarray]] = []
        for idx, name in enumerate(row["channel_names"]):
            channel = values[idx]
            unit = row["units"][idx]
            unit_str = f" in {unit}" if unit else ""
            prompts.append(
                self._continuous_prompt(
                    channel,
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


def _majority_labels_per_second(label_values: Iterable[int]) -> np.ndarray:
    labels = np.asarray(label_values, dtype=np.int16).reshape(-1)
    n_seconds = int(np.ceil(len(labels) / _LABEL_SAMPLE_RATE_HZ))
    result = np.zeros(n_seconds, dtype=np.int16)
    for second in range(n_seconds):
        start = second * _LABEL_SAMPLE_RATE_HZ
        end = min(start + _LABEL_SAMPLE_RATE_HZ, len(labels))
        if start >= end:
            continue
        counts = np.bincount(labels[start:end].astype(np.int64), minlength=8)
        result[second] = int(np.argmax(counts))
    return result


def _window_majority_label(
    labels: np.ndarray,
    allowed_labels: dict[int, str],
) -> tuple[int | None, float]:
    if labels.size == 0:
        return None, 0.0
    counts = np.bincount(labels.astype(np.int64), minlength=max(allowed_labels) + 1)
    label = int(np.argmax(counts))
    purity = float(counts[label] / labels.size)
    if label not in allowed_labels:
        return None, purity
    return label, purity
