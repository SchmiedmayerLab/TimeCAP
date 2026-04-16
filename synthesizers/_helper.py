#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np

from extractors import ChannelConfig
from mhc.constants import WATCH_HR_CHANNEL
from timef.schema import Recording


def index_or_none(row: Recording, channel_name: str) -> int | None:
    try:
        return row.channel_names.index(channel_name)
    except ValueError:
        return None


def positive_metric_values(row: Recording, idx: int | None, start: int, end: int) -> np.ndarray | None:
    if idx is None:
        return None
    values = np.asarray(row.values[idx][start:end], dtype=float)
    valid = np.isfinite(values) & (values > 0)
    if not valid.any():
        return None
    return values[valid]


def metric_mean(row: Recording, idx: int | None, start: int, end: int) -> float | None:
    values = positive_metric_values(row, idx, start, end)
    if values is None:
        return None
    return float(np.mean(values))


def metric_peak(row: Recording, idx: int | None, start: int, end: int) -> float | None:
    values = positive_metric_values(row, idx, start, end)
    if values is None:
        return None
    return float(np.max(values))


def metric_total(row: Recording, idx: int | None, start: int, end: int) -> float | None:
    values = positive_metric_values(row, idx, start, end)
    if values is None:
        return None
    return float(np.sum(values))


def metric_day_mean_delta(row: Recording, idx: int | None, start: int, end: int) -> float | None:
    window_mean = metric_mean(row, idx, start, end)
    if window_mean is None or idx is None:
        return None

    day_values = positive_metric_values(row, idx, 0, row.values.shape[1])
    if day_values is None:
        return None
    return float(window_mean - np.mean(day_values))


def metric_day_mean(row: Recording, idx: int | None) -> float | None:
    if idx is None:
        return None
    day_values = positive_metric_values(row, idx, 0, row.values.shape[1])
    if day_values is None:
        return None
    return float(np.mean(day_values))


def channel_meta(config: ChannelConfig, channel_name: str) -> tuple[str, str]:
    display_name, unit, _ = config.meta.get(channel_name, (config.display_name(channel_name), "", 0))
    return display_name, unit


def format_metric_summary(
    config: ChannelConfig,
    channel_name: str,
    mean: float,
    peak: float | None = None,
    elevated_threshold: float | None = None,
) -> str:
    display_name, unit = channel_meta(config, channel_name)
    summary = f"averaging a {display_name} of {mean:.0f} {unit}"
    if peak is not None:
        summary += f", peaking at {peak:.0f} {unit}"
    if elevated_threshold is not None and mean > elevated_threshold:
        summary += f", with an elevated {display_name} during this phase"
    return summary


def sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return text if text.endswith(".") else f"{text}."


def finalize_caption_text(text: str, metrics_suffix: str) -> str:
    if metrics_suffix:
        return text.rstrip(".") + "."
    return text


def append_hr_metrics(
    parts: list[str],
    channel_idxs: list[int],
    config: ChannelConfig,
    row: Recording,
    start: int,
    end: int,
    hr_idx: int | None,
    elevated_threshold: float,
    include_space_before_day_unit: bool = True,
) -> None:
    """Append heart-rate summary sentences and include the HR channel when present."""
    hr_mean = metric_mean(row, hr_idx, start, end)
    if hr_mean is None or hr_idx is None:
        return

    hr_peak = metric_peak(row, hr_idx, start, end)
    parts.append(
        sentence(
            format_metric_summary(
                config=config,
                channel_name=WATCH_HR_CHANNEL,
                mean=hr_mean,
                peak=hr_peak,
                elevated_threshold=elevated_threshold,
            )
        )
    )

    hr_day_delta = metric_day_mean_delta(row, hr_idx, start, end)
    hr_day_mean = metric_day_mean(row, hr_idx)
    if hr_day_delta is not None and hr_day_mean is not None:
        hr_name, hr_unit = channel_meta(config, WATCH_HR_CHANNEL)
        direction = "higher" if hr_day_delta >= 0 else "lower"
        day_mean_unit = f" {hr_unit}" if include_space_before_day_unit and hr_unit else hr_unit
        parts.append(
            sentence(
                f"The {hr_name} was {abs(hr_day_delta):.0f} {hr_unit} {direction} than the day's mean of {hr_day_mean:.0f}{day_mean_unit}"
            )
        )

    channel_idxs.append(hr_idx)


def append_distance_metrics(
    parts: list[str],
    channel_idxs: list[int],
    row: Recording,
    start: int,
    end: int,
    distance_idx: int | None,
) -> None:
    """Append distance summary sentences and include the distance channel when present."""
    distance_mean = metric_mean(row, distance_idx, start, end)
    distance_total = metric_total(row, distance_idx, start, end)
    if distance_mean is None or distance_idx is None:
        return

    parts.append(sentence(f"The watch recorded an average distance of {distance_mean:.1f} m/min during this period"))
    if distance_total is not None:
        parts.append(sentence(f"The total distance recorded by the watch in that interval was {distance_total:.1f} m"))

    channel_idxs.append(distance_idx)


def append_step_metrics(
    parts: list[str],
    channel_idxs: list[int],
    row: Recording,
    start: int,
    end: int,
    step_idx: int | None,
) -> None:
    """Append step-count summary sentences and include the step channel when present."""
    step_mean = metric_mean(row, step_idx, start, end)
    step_total = metric_total(row, step_idx, start, end)
    if step_mean is None or step_idx is None:
        return

    parts.append(sentence(f"The watch recorded an average step count of {step_mean:.1f} steps/min during this period"))
    if step_total is not None:
        parts.append(sentence(f"The total step count recorded by the watch during that time was {step_total:.0f}"))

    channel_idxs.append(step_idx)


def contiguous_windows(mask: np.ndarray, min_duration: int) -> list[tuple[int, int]]:
    if not mask.any():
        return []

    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    keep = (ends - starts) >= min_duration
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))
