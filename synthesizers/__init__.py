#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import abc

import numpy as np

from extractors import ChannelConfig
from timef.schema import Annotation, Recording


class CrossChannelSynthesizer(abc.ABC):
    @abc.abstractmethod
    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]: ...

    @staticmethod
    def _index_or_none(row: Recording, channel_name: str) -> int | None:
        try:
            return row.channel_names.index(channel_name)
        except ValueError:
            return None

    @staticmethod
    def _positive_metric_values(row: Recording, idx: int | None, start: int, end: int) -> np.ndarray | None:
        if idx is None:
            return None
        values = np.asarray(row.values[idx][start:end], dtype=float)
        valid = np.isfinite(values) & (values > 0)
        if not valid.any():
            return None
        return values[valid]

    @classmethod
    def _metric_mean(cls, row: Recording, idx: int | None, start: int, end: int) -> float | None:
        values = cls._positive_metric_values(row, idx, start, end)
        if values is None:
            return None
        return float(np.mean(values))

    @classmethod
    def _metric_peak(cls, row: Recording, idx: int | None, start: int, end: int) -> float | None:
        values = cls._positive_metric_values(row, idx, start, end)
        if values is None:
            return None
        return float(np.max(values))

    @classmethod
    def _metric_total(cls, row: Recording, idx: int | None, start: int, end: int) -> float | None:
        values = cls._positive_metric_values(row, idx, start, end)
        if values is None:
            return None
        return float(np.sum(values))

    @classmethod
    def _metric_day_mean_delta(cls, row: Recording, idx: int | None, start: int, end: int) -> float | None:
        window_mean = cls._metric_mean(row, idx, start, end)
        if window_mean is None or idx is None:
            return None

        day_values = cls._positive_metric_values(row, idx, 0, row.values.shape[1])
        if day_values is None:
            return None
        return float(window_mean - np.mean(day_values))

    @classmethod
    def _metric_day_mean(cls, row: Recording, idx: int | None) -> float | None:
        if idx is None:
            return None
        day_values = cls._positive_metric_values(row, idx, 0, row.values.shape[1])
        if day_values is None:
            return None
        return float(np.mean(day_values))

    @staticmethod
    def _channel_meta(config: ChannelConfig, channel_name: str) -> tuple[str, str]:
        display_name, unit, _ = config.meta.get(channel_name, (config.display_name(channel_name), "", 0))
        return display_name, unit

    @classmethod
    def _format_metric_summary(
        cls,
        config: ChannelConfig,
        channel_name: str,
        mean: float,
        peak: float | None = None,
        elevated_threshold: float | None = None,
    ) -> str:
        display_name, unit = cls._channel_meta(config, channel_name)
        summary = f"averaging a {display_name} of {mean:.0f} {unit}"
        if peak is not None:
            summary += f", peaking at {peak:.0f} {unit}"
        if elevated_threshold is not None and mean > elevated_threshold:
            summary += f", with an elevated {display_name} during this phase"
        return summary

    @staticmethod
    def _sentence(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        return text if text.endswith(".") else f"{text}."

    @staticmethod
    def _finalize_caption_text(text: str, metrics_suffix: str) -> str:
        if metrics_suffix:
            return text.rstrip(".") + "."
        return text


def contiguous_windows(mask: np.ndarray, min_duration: int) -> list[tuple[int, int]]:
    if not mask.any():
        return []

    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    keep = (ends - starts) >= min_duration
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))
