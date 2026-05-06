#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json

import numpy as np

from extractors import ChannelConfig
from synthesizers import CrossChannelSynthesizer
from synthesizers._helper import (
    append_low_variability_metrics,
    append_hr_metrics,
    append_relative_level_metrics,
    contiguous_windows,
    finalize_caption_text,
    index_or_none,
    metric_day_mean_delta,
    render_metric_text,
    sentence,
)
from timef.schema import Annotation, Recording
from util import seed_from_key
from wesad.constants import (
    WESAD_CHEST_ACC_MAG_CHANNEL,
    WESAD_CHEST_EDA_CHANNEL,
    WESAD_CHEST_RESP_CHANNEL,
    WESAD_HR_CHANNEL,
    WESAD_WRIST_ACC_MAG_CHANNEL,
    WESAD_WRIST_EDA_CHANNEL,
)


class _WESADLabelWindowSynthesizer(CrossChannelSynthesizer):
    label_channel: str
    annotation_label: str
    template_key: str

    def __init__(self, min_duration: int = 30):
        self.min_duration = min_duration

    def synthesize(self, row: Recording, config: ChannelConfig) -> list[Annotation]:
        try:
            label_idx = row.channel_names.index(self.label_channel)
        except ValueError:
            return []

        labels = np.asarray(row.values[label_idx], dtype=float)
        active = (labels > 0) & ~np.isnan(labels)
        if not active.any():
            return []

        payload = json.loads(config.templates_path.read_text())
        templates = payload["cross_channel"][self.template_key]
        time_unit = "second" if config.time_unit == "seconds" else config.time_unit.rstrip("s")
        seed = seed_from_key(f"{row.row_id}:{self.annotation_label}")

        results: list[Annotation] = []
        for i, (start, end) in enumerate(contiguous_windows(active, self.min_duration)):
            end_inclusive = max(start, end - 1)
            template = templates[(seed + i) % len(templates)]
            channel_idxs = [label_idx]
            metrics_suffix = self._metrics_suffix(config, row, start, end, channel_idxs)
            text = template.format(
                time_unit=time_unit,
                start=start,
                end=end_inclusive,
                metrics_suffix=metrics_suffix,
            )
            text = finalize_caption_text(text, metrics_suffix)
            results.append(
                Annotation(
                    caption_type="cross_channel",
                    text=text,
                    channel_idxs=tuple(dict.fromkeys(channel_idxs)),
                    window=(start, end),
                    label=self.annotation_label,
                )
            )
        return results

    def _metrics_suffix(
        self,
        config: ChannelConfig,
        row: Recording,
        start: int,
        end: int,
        channel_idxs: list[int],
    ) -> str:
        raise NotImplementedError


class StressResponseSynthesizer(_WESADLabelWindowSynthesizer):
    label_channel = "wesad:label:stress"
    annotation_label = "stress_response"
    template_key = "stress_response"

    def _metrics_suffix(
        self,
        config: ChannelConfig,
        row: Recording,
        start: int,
        end: int,
        channel_idxs: list[int],
    ) -> str:
        parts: list[str] = []

        hr_idx = index_or_none(row, WESAD_HR_CHANNEL)
        hr_delta = metric_day_mean_delta(row, hr_idx, start, end) if hr_idx is not None else None
        if hr_delta is not None and hr_delta >= 3.0:
            append_hr_metrics(parts, channel_idxs, config, row, start, end, hr_idx, elevated_threshold=100.0)

        _append_best_relative_metric(
            parts,
            channel_idxs,
            config,
            row,
            start,
            end,
            candidate_channels=[WESAD_CHEST_EDA_CHANNEL, WESAD_WRIST_EDA_CHANNEL],
            template_key="eda_higher",
            fallback="{name} was higher than the session baseline in this interval",
            direction="higher",
            min_delta=0.03,
            std_scale=0.35,
        )
        _append_best_relative_metric(
            parts,
            channel_idxs,
            config,
            row,
            start,
            end,
            candidate_channels=[WESAD_CHEST_RESP_CHANNEL],
            template_key="resp_higher",
            fallback="{name} was stronger than the session baseline in this interval",
            direction="higher",
            min_delta=0.10,
            std_scale=0.45,
        )
        _append_best_relative_metric(
            parts,
            channel_idxs,
            config,
            row,
            start,
            end,
            candidate_channels=[WESAD_CHEST_ACC_MAG_CHANNEL, WESAD_WRIST_ACC_MAG_CHANNEL],
            template_key="movement_higher",
            fallback="{name} was higher than the session baseline in this interval",
            direction="higher",
            min_delta=0.05,
            std_scale=0.50,
        )

        if not parts:
            return ""
        return " " + " ".join(parts)


class RecoveryCalmSynthesizer(_WESADLabelWindowSynthesizer):
    def __init__(self, label_channel: str, annotation_label: str, template_key: str, min_duration: int = 30):
        super().__init__(min_duration=min_duration)
        self.label_channel = label_channel
        self.annotation_label = annotation_label
        self.template_key = template_key

    def _metrics_suffix(
        self,
        config: ChannelConfig,
        row: Recording,
        start: int,
        end: int,
        channel_idxs: list[int],
    ) -> str:
        parts: list[str] = []

        hr_idx = index_or_none(row, WESAD_HR_CHANNEL)
        hr_delta = metric_day_mean_delta(row, hr_idx, start, end) if hr_idx is not None else None
        if hr_delta is not None and hr_delta <= 3.0:
            append_hr_metrics(parts, channel_idxs, config, row, start, end, hr_idx, elevated_threshold=100.0)

        _append_best_relative_metric(
            parts,
            channel_idxs,
            config,
            row,
            start,
            end,
            candidate_channels=[WESAD_CHEST_EDA_CHANNEL, WESAD_WRIST_EDA_CHANNEL],
            template_key="eda_lower",
            fallback="{name} stayed below the session baseline in this interval",
            direction="lower",
            min_delta=0.02,
            std_scale=0.25,
        )
        _append_best_low_variability_metric(
            parts,
            channel_idxs,
            config,
            row,
            start,
            end,
            candidate_channels=[WESAD_CHEST_RESP_CHANNEL],
            template_key="resp_steady",
            fallback="{name} was steady during this interval",
            max_std_ratio=0.70,
        )
        _append_best_relative_metric(
            parts,
            channel_idxs,
            config,
            row,
            start,
            end,
            candidate_channels=[WESAD_CHEST_ACC_MAG_CHANNEL, WESAD_WRIST_ACC_MAG_CHANNEL],
            template_key="movement_lower",
            fallback="{name} stayed below the session baseline in this interval",
            direction="lower",
            min_delta=0.03,
            std_scale=0.25,
        )
        _append_best_low_variability_metric(
            parts,
            channel_idxs,
            config,
            row,
            start,
            end,
            candidate_channels=[WESAD_CHEST_EDA_CHANNEL, WESAD_WRIST_EDA_CHANNEL],
            template_key="eda_flat",
            fallback="{name} remained comparatively flat in this interval",
            max_std_ratio=0.65,
        )

        if not parts:
            return ""
        return " " + " ".join(parts)


class AmusementActivationSynthesizer(_WESADLabelWindowSynthesizer):
    label_channel = "wesad:label:amusement"
    annotation_label = "amusement_activation"
    template_key = "amusement_activation"

    def _metrics_suffix(
        self,
        config: ChannelConfig,
        row: Recording,
        start: int,
        end: int,
        channel_idxs: list[int],
    ) -> str:
        parts: list[str] = []

        hr_idx = index_or_none(row, WESAD_HR_CHANNEL)
        hr_delta = metric_day_mean_delta(row, hr_idx, start, end) if hr_idx is not None else None
        if hr_delta is not None and hr_delta >= 1.5:
            append_hr_metrics(parts, channel_idxs, config, row, start, end, hr_idx, elevated_threshold=100.0)

        movement_found = _append_best_relative_metric(
            parts,
            channel_idxs,
            config,
            row,
            start,
            end,
            candidate_channels=[WESAD_CHEST_ACC_MAG_CHANNEL, WESAD_WRIST_ACC_MAG_CHANNEL],
            template_key="movement_higher",
            fallback="{name} was higher than the session baseline in this interval",
            direction="higher",
            min_delta=0.04,
            std_scale=0.35,
        )
        eda_found = _append_best_relative_metric(
            parts,
            channel_idxs,
            config,
            row,
            start,
            end,
            candidate_channels=[WESAD_CHEST_EDA_CHANNEL, WESAD_WRIST_EDA_CHANNEL],
            template_key="eda_higher",
            fallback="{name} was higher than the session baseline in this interval",
            direction="higher",
            min_delta=0.02,
            std_scale=0.25,
        )

        if movement_found and not eda_found and hr_delta is not None and hr_delta < 8.0:
            parts.append(
                sentence(
                    render_metric_text(
                        config,
                        row,
                        start,
                        end,
                        "amusement_not_stress_like",
                        "Physiological activation remained milder than a clearly stress-like pattern.",
                    )
                )
            )

        if not parts:
            return ""
        return " " + " ".join(parts)


def default_wesad_synthesizers() -> list[CrossChannelSynthesizer]:
    return [
        StressResponseSynthesizer(min_duration=30),
        RecoveryCalmSynthesizer(
            label_channel="wesad:label:baseline",
            annotation_label="calm_baseline",
            template_key="calm_baseline",
            min_duration=30,
        ),
        RecoveryCalmSynthesizer(
            label_channel="wesad:label:meditation",
            annotation_label="calm_meditation",
            template_key="calm_meditation",
            min_duration=30,
        ),
        AmusementActivationSynthesizer(min_duration=30),
    ]


def _append_best_relative_metric(
    parts: list[str],
    channel_idxs: list[int],
    config: ChannelConfig,
    row: Recording,
    start: int,
    end: int,
    candidate_channels: list[str],
    template_key: str,
    fallback: str,
    direction: str,
    min_delta: float,
    std_scale: float,
) -> bool:
    best = _select_best_channel(row, start, end, candidate_channels, direction)
    if best is None:
        return False
    idx = index_or_none(row, best)
    return append_relative_level_metrics(
        parts,
        channel_idxs,
        config,
        row,
        start,
        end,
        idx,
        best,
        template_key=template_key,
        fallback=fallback,
        direction=direction,
        min_delta=min_delta,
        std_scale=std_scale,
    )


def _append_best_low_variability_metric(
    parts: list[str],
    channel_idxs: list[int],
    config: ChannelConfig,
    row: Recording,
    start: int,
    end: int,
    candidate_channels: list[str],
    template_key: str,
    fallback: str,
    max_std_ratio: float,
) -> bool:
    best = _select_best_channel(row, start, end, candidate_channels, "lower")
    if best is None:
        return False
    idx = index_or_none(row, best)
    return append_low_variability_metrics(
        parts,
        channel_idxs,
        config,
        row,
        start,
        end,
        idx,
        best,
        template_key=template_key,
        fallback=fallback,
        max_std_ratio=max_std_ratio,
    )


def _select_best_channel(
    row: Recording,
    start: int,
    end: int,
    candidate_channels: list[str],
    direction: str,
) -> str | None:
    best_name: str | None = None
    best_score: float | None = None
    for channel_name in candidate_channels:
        idx = index_or_none(row, channel_name)
        if idx is None:
            continue
        delta = metric_day_mean_delta(row, idx, start, end)
        if delta is None:
            continue
        score = float(delta if direction == "higher" else -delta)
        if best_score is None or score > best_score:
            best_name = channel_name
            best_score = score
    return best_name
