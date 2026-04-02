#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import abc
import re
import zlib
from dataclasses import dataclass, field

from aggregators import MetricAggregator
from detectors import StructuralDetector
from timef.schema import Annotation, AnnotationSpec, Sample, SampleRef, SampleSignalRef, Signal

VALID_CAPTION_TYPES = ("statistical", "structural", "semantic")


_ACTIVITY_RE = re.compile(r"HKWorkoutActivityType(.+)$")


@dataclass(frozen=True)
class ChannelConfig:
    names: list[str]                                    # ordered channel names
    meta: dict[str, tuple[str, str, int]]               # channel -> (display_name, unit, decimals)
    continuous: frozenset[str]                           # which channels are continuous
    aggregators: dict[str, MetricAggregator] = field(default_factory=dict)  # channel -> aggregator override
    detectors: dict[str, list[StructuralDetector]] = field(default_factory=dict)  # channel -> detector list

    def display_name(self, channel: str) -> str:
        if channel in self.meta:
            return self.meta[channel][0]
        m = _ACTIVITY_RE.search(channel)
        if m:
            return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", m.group(1)).lower()
        if ":" in channel:
            return channel.split(":", 1)[1]
        return channel


class CaptionExtractor(abc.ABC):
    caption_type: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            _validate_caption_type(cls)

    def __init__(self, config: ChannelConfig):
        self.config = config

    def get_annotation_spec(self) -> AnnotationSpec:
        return AnnotationSpec(id=f"captioning:{self.caption_type}", task="captioning")

    @staticmethod
    def _seed(key: str) -> int:
        return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF

    @abc.abstractmethod
    def extract(self, signals: list[Signal]) -> list[tuple[Sample, Annotation]]:
        """Extract captions and return (Sample, Annotation) pairs."""
        ...


def _validate_caption_type(cls):
    ct = getattr(cls, "caption_type", None)
    if ct is None:
        raise TypeError(f"{cls.__name__} must define class attribute 'caption_type'.")
    if ct not in VALID_CAPTION_TYPES:
        raise TypeError(f"{cls.__name__}.caption_type must be one of {VALID_CAPTION_TYPES!r}, got {ct!r}.")
