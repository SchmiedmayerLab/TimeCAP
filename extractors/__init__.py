#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import abc
import pathlib
import re
from dataclasses import dataclass, field

from aggregators import MetricAggregator
from detectors import StructuralDetector
from timef.schema import Annotation, Recording
from util import seed_from_key

DEFAULT_TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent.parent / "templates" / "templates.json"

VALID_CAPTION_TYPES = ("statistical", "structural", "semantic", "cross_channel")


_ACTIVITY_RE = re.compile(r"HKWorkoutActivityType(.+)$")


@dataclass(frozen=True)
class ChannelConfig:
    names: list[str]                                    # ordered channel names
    meta: dict[str, tuple[str, str, int]]               # channel -> (display_name, unit, decimals)
    continuous: frozenset[str]                           # which channels are continuous
    groups: dict[str, frozenset[str]] = field(default_factory=dict)  # semantic/logical channel groups
    aggregators: dict[str, MetricAggregator] = field(default_factory=dict)  # channel -> aggregator override
    detectors: dict[str, list[StructuralDetector]] = field(default_factory=dict)  # channel -> detector list
    templates_path: pathlib.Path = DEFAULT_TEMPLATES_PATH
    time_unit: str = "minutes"                           # label for the time axis (e.g. "minutes", "hours")

    def display_name(self, channel: str) -> str:
        if channel in self.meta:
            return self.meta[channel][0]
        m = _ACTIVITY_RE.search(channel)
        if m:
            suffix = m.group(1)
            if suffix == "Other":
                return "other activity"
            return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", suffix).lower()
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

    @staticmethod
    def _seed(key: str) -> int:
        return seed_from_key(key)

    @abc.abstractmethod
    def extract(self, row: Recording) -> list[Annotation]:
        """Extract annotations for a row."""
        ...


def _validate_caption_type(cls):
    ct = getattr(cls, "caption_type", None)
    if ct is None:
        raise TypeError(f"{cls.__name__} must define class attribute 'caption_type'.")
    if ct not in VALID_CAPTION_TYPES:
        raise TypeError(f"{cls.__name__}.caption_type must be one of {VALID_CAPTION_TYPES!r}, got {ct!r}.")
