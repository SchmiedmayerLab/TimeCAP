#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from extractors import ChannelConfig
from mhc.constants import WATCH_ACTIVE_ENERGY_CHANNEL, WATCH_HR_CHANNEL
from timef.schema import Recording
from util import capitalize, seed_from_key


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


def rate_unit(config: ChannelConfig, channel_name: str, fallback: str) -> str:
    _, unit = channel_meta(config, channel_name)
    return unit or fallback


@lru_cache(maxsize=None)
def _load_template_payload(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cross_channel_metric_templates(config: ChannelConfig) -> dict[str, list[str]]:
    payload = _load_template_payload(str(config.templates_path))
    cross_channel = payload.get("cross_channel", {})
    metrics = cross_channel.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _select_metric_template(
    config: ChannelConfig,
    row: Recording,
    start: int,
    end: int,
    template_key: str,
) -> str | None:
    templates = _cross_channel_metric_templates(config).get(template_key)
    if not templates:
        return None
    idx = seed_from_key(f"{row.row_id}:{start}:{end}:{template_key}") % len(templates)
    return templates[idx]


def render_metric_text(
    config: ChannelConfig,
    row: Recording,
    start: int,
    end: int,
    template_key: str,
    fallback: str,
    **kwargs,
) -> str:
    template = _select_metric_template(config, row, start, end, template_key)
    return (template or fallback).format(**kwargs)


def sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return text if text.endswith(".") else f"{text}."


def finalize_caption_text(text: str, metrics_suffix: str) -> str:
    if metrics_suffix:
        text = text.rstrip(".") + "."
    return capitalize(text)


def _join_metric_fragments(fragments: list[str]) -> str:
    return ", ".join(fragment.strip() for fragment in fragments if fragment.strip())


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

    hr_name, hr_unit = channel_meta(config, WATCH_HR_CHANNEL)
    hr_peak = metric_peak(row, hr_idx, start, end)
    summary_parts = [
        render_metric_text(
            config,
            row,
            start,
            end,
            "hr_summary",
            "averaging a {name} of {mean:.0f} {unit}",
            name=hr_name,
            mean=hr_mean,
            unit=hr_unit,
        )
    ]
    if hr_peak is not None and round(hr_peak) != round(hr_mean):
        summary_parts.append(
            render_metric_text(
                config,
                row,
                start,
                end,
                "hr_peak",
                "peaking at {peak:.0f} {unit}",
                name=hr_name,
                peak=hr_peak,
                unit=hr_unit,
            )
        )
    if elevated_threshold is not None and hr_mean > elevated_threshold:
        summary_parts.append(
            render_metric_text(
                config,
                row,
                start,
                end,
                "hr_elevated",
                "with an elevated {name} during this phase",
                name=hr_name,
            )
        )
    parts.append(sentence(_join_metric_fragments(summary_parts)))

    hr_day_delta = metric_day_mean_delta(row, hr_idx, start, end)
    hr_day_mean = metric_day_mean(row, hr_idx)
    if hr_day_delta is not None and hr_day_mean is not None:
        close_threshold = 3.0
        if abs(hr_day_delta) < close_threshold:
            parts.append(
                sentence(
                    render_metric_text(
                        config,
                        row,
                        start,
                        end,
                        "hr_day_delta_close",
                        "The {name} stayed close to the day's mean of {day_mean:.0f}{day_mean_unit}",
                        name=hr_name,
                        day_mean=hr_day_mean,
                        day_mean_unit=f" {hr_unit}" if include_space_before_day_unit and hr_unit else hr_unit,
                    )
                )
            )
            channel_idxs.append(hr_idx)
            return

        direction = "higher" if hr_day_delta >= 0 else "lower"
        day_mean_unit = f" {hr_unit}" if include_space_before_day_unit and hr_unit else hr_unit
        template_key = "hr_day_delta_higher" if hr_day_delta >= 0 else "hr_day_delta_lower"
        fallback = "The {name} was {delta:.0f} {unit} {direction} than the day's mean of {day_mean:.0f}{day_mean_unit}"
        parts.append(
            sentence(
                render_metric_text(
                    config,
                    row,
                    start,
                    end,
                    template_key,
                    fallback,
                    name=hr_name,
                    delta=abs(hr_day_delta),
                    unit=hr_unit,
                    direction=direction,
                    day_mean=hr_day_mean,
                    day_mean_unit=day_mean_unit,
                )
            )
        )

    channel_idxs.append(hr_idx)


def append_relative_level_metrics(
    parts: list[str],
    channel_idxs: list[int],
    config: ChannelConfig,
    row: Recording,
    start: int,
    end: int,
    idx: int | None,
    channel_name: str,
    template_key: str,
    fallback: str,
    direction: str,
    min_delta: float,
    std_scale: float,
) -> bool:
    """Append a relative-level sentence when a channel deviates from the session baseline."""
    if idx is None:
        return False

    day_mean = metric_day_mean(row, idx)
    window_mean = metric_mean(row, idx, start, end)
    if day_mean is None or window_mean is None:
        return False

    delta = window_mean - day_mean
    if direction == "higher":
        if delta < min_delta:
            return False
    else:
        if delta > -min_delta:
            return False

    std = float(np.nanstd(np.asarray(row.values[idx][start:end], dtype=float))) if end > start else 0.0
    day_std = float(np.nanstd(np.asarray(row.values[idx], dtype=float)))
    if not np.isfinite(day_std) or day_std <= 0:
        day_std = 1.0
    if std > day_std * std_scale:
        return False

    display_name, unit = channel_meta(config, channel_name)
    parts.append(
        sentence(
            render_metric_text(
                config,
                row,
                start,
                end,
                template_key,
                fallback,
                name=display_name,
                mean=window_mean,
                day_mean=day_mean,
                delta=abs(delta),
                unit=unit,
                direction=direction,
            )
        )
    )
    channel_idxs.append(idx)
    return True


def append_low_variability_metrics(
    parts: list[str],
    channel_idxs: list[int],
    config: ChannelConfig,
    row: Recording,
    start: int,
    end: int,
    idx: int | None,
    channel_name: str,
    template_key: str,
    fallback: str,
    max_std_ratio: float,
) -> bool:
    """Append a low-variability sentence when the interval is calmer than the session baseline."""
    if idx is None:
        return False

    values = np.asarray(row.values[idx], dtype=float)
    window = values[start:end]
    if window.size == 0:
        return False
    if not np.isfinite(window).any():
        return False

    window_std = float(np.nanstd(window))
    day_std = float(np.nanstd(values))
    if not np.isfinite(day_std) or day_std <= 0:
        return False
    if window_std > day_std * max_std_ratio:
        return False

    display_name, _unit = channel_meta(config, channel_name)
    parts.append(
        sentence(
            render_metric_text(
                config,
                row,
                start,
                end,
                template_key,
                fallback,
                name=display_name,
                std=window_std,
                day_std=day_std,
            )
        )
    )
    channel_idxs.append(idx)
    return True


def append_active_energy_metrics(
    parts: list[str],
    channel_idxs: list[int],
    config: ChannelConfig,
    row: Recording,
    start: int,
    end: int,
    energy_idx: int | None,
) -> None:
    """Append active-energy summary sentences and include the energy channel when present."""
    energy_mean = metric_mean(row, energy_idx, start, end)
    energy_total = metric_total(row, energy_idx, start, end)
    if energy_mean is None or energy_idx is None:
        return

    energy_name, energy_unit = channel_meta(config, WATCH_ACTIVE_ENERGY_CHANNEL)
    parts.append(
        sentence(
            render_metric_text(
                config,
                row,
                start,
                end,
                "active_energy_mean",
                "The watch recorded an average {name} of {mean:.0f} {unit} during this period",
                name=energy_name,
                mean=energy_mean,
                unit=energy_unit,
            )
        )
    )
    if energy_total is not None:
        parts.append(
            sentence(
                render_metric_text(
                    config,
                    row,
                    start,
                    end,
                    "active_energy_total",
                    "The total {name} recorded by the watch in that interval was {total:.0f} {unit}",
                    name=energy_name,
                    total=energy_total,
                    unit=energy_unit.replace("/min", "").replace("/hr", ""),
                )
            )
        )

    channel_idxs.append(energy_idx)


def append_distance_metrics(
    parts: list[str],
    channel_idxs: list[int],
    config: ChannelConfig,
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
    distance_rate_unit = rate_unit(config, row.channel_names[distance_idx], "m/min")

    parts.append(
        sentence(
            render_metric_text(
                config,
                row,
                start,
                end,
                "distance_mean",
                "The watch recorded an average distance of {mean:.0f} {rate_unit} during this period",
                mean=distance_mean,
                rate_unit=distance_rate_unit,
            )
        )
    )
    if distance_total is not None:
        parts.append(
            sentence(
                render_metric_text(
                    config,
                    row,
                    start,
                    end,
                    "distance_total",
                    "The total distance recorded by the watch in that interval was {total:.0f} m",
                    total=distance_total,
                )
            )
        )

    channel_idxs.append(distance_idx)


def append_step_metrics(
    parts: list[str],
    channel_idxs: list[int],
    config: ChannelConfig,
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
    step_rate_unit = rate_unit(config, row.channel_names[step_idx], "steps/min")

    parts.append(
        sentence(
            render_metric_text(
                config,
                row,
                start,
                end,
                "step_mean",
                "The watch recorded an average step count of {mean:.0f} {rate_unit} during this period",
                mean=step_mean,
                rate_unit=step_rate_unit,
            )
        )
    )
    if step_total is not None:
        parts.append(
            sentence(
                render_metric_text(
                    config,
                    row,
                    start,
                    end,
                    "step_total",
                    "The total step count recorded by the watch during that time was {total:.0f}",
                    total=step_total,
                )
            )
        )

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
