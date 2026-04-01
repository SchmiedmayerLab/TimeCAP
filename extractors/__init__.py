#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import abc
import zlib
from dataclasses import dataclass, field

from aggregators import MetricAggregator
from detectors import StructuralDetector
from timef.runtime import CaptionData, SignalData
from timef.schema import AnnotationSpec

VALID_CAPTION_TYPES = ("statistical", "structural", "semantic")
CAPTION_SPEC_IDS = {name: idx for idx, name in enumerate(VALID_CAPTION_TYPES)}


@dataclass(frozen=True)
class ChannelConfig:
    names: list[str]                                    # ordered channel names
    meta: dict[str, tuple[str, str, int]]               # channel -> (display_name, unit, decimals)
    continuous: frozenset[str]                           # which channels are continuous
    aggregators: dict[str, MetricAggregator] = field(default_factory=dict)  # channel -> aggregator override
    detectors: dict[str, list[StructuralDetector]] = field(default_factory=dict)  # channel -> detector list


class CaptionExtractor(abc.ABC):
    caption_type: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            _validate_caption_type(cls)

    def __init__(self, config: ChannelConfig):
        self.config = config

    def get_annotation_spec(self) -> AnnotationSpec:
        return AnnotationSpec(id=CAPTION_SPEC_IDS[self.caption_type], task="captioning")

    @staticmethod
    def _seed(key: str) -> int:
        return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF

    @abc.abstractmethod
    def extract(self, signals: list[SignalData]) -> list[CaptionData]:
        """Extract captions from signals and return lightweight CaptionData records."""
        ...


def _validate_caption_type(cls):
    ct = getattr(cls, "caption_type", None)
    if ct is None:
        raise TypeError(f"{cls.__name__} must define class attribute 'caption_type'.")
    if ct not in VALID_CAPTION_TYPES:
        raise TypeError(f"{cls.__name__}.caption_type must be one of {VALID_CAPTION_TYPES!r}, got {ct!r}.")
