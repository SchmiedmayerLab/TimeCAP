#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generator

import numpy as np


@dataclass(slots=True)
class Annotation:
    caption_type: str
    text: str
    channel_idxs: tuple[int, ...] = ()
    window: tuple[int, int] | None = None
    label: Any = None


@dataclass(slots=True)
class SignalView:
    row: "Recording"
    idx: int

    @property
    def id(self) -> str:
        return f"{self.row.row_id}:{self.name}"

    @property
    def name(self) -> str:
        return self.row.channel_names[self.idx]

    @property
    def display_name(self) -> str:
        return self.row.display_names[self.idx]

    @property
    def unit(self) -> str | None:
        return self.row.units[self.idx]

    @property
    def data(self) -> np.ndarray:
        return self.row.values[self.idx]

    @property
    def has_any_data(self) -> bool:
        return bool(self.row.has_any_data[self.idx])

    @property
    def minutes_nonzero_or_nan(self) -> float | None:
        if self.row.minutes_nonzero_or_nan is None:
            return None
        return float(self.row.minutes_nonzero_or_nan[self.idx])

    @property
    def variance(self) -> float:
        return float(self.row.channel_variance[self.idx])


@dataclass(slots=True)
class Recording:
    row_id: str
    user_id: str
    date: str
    values: np.ndarray
    channel_names: tuple[str, ...]
    display_names: tuple[str, ...]
    units: tuple[str | None, ...]
    has_any_data: np.ndarray
    channel_variance: np.ndarray
    minutes_nonzero_or_nan: np.ndarray | None = None
    total_nonwear_minutes: float | None = None
    wear_pct: float | None = None
    annotations: list[Annotation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("values must have shape (channels, time)")
        n_channels = self.values.shape[0]
        for field_name in ("channel_names", "display_names", "units"):
            if len(getattr(self, field_name)) != n_channels:
                raise ValueError(f"{field_name} length must match number of channels")
        for field_name in ("has_any_data", "channel_variance"):
            if getattr(self, field_name).shape != (n_channels,):
                raise ValueError(f"{field_name} must have shape ({n_channels},)")
        if self.minutes_nonzero_or_nan is not None and self.minutes_nonzero_or_nan.shape != (n_channels,):
            raise ValueError(f"minutes_nonzero_or_nan must have shape ({n_channels},)")

    def signal(self, idx: int) -> SignalView:
        return SignalView(row=self, idx=idx)

    def iter_channels(self) -> Generator[SignalView, None, None]:
        for idx in range(self.values.shape[0]):
            yield SignalView(row=self, idx=idx)

    def channel(self, idx: int) -> SignalView:
        return self.signal(idx)

    def annotations_for_signal(self, idx: int) -> list[Annotation]:
        return [annotation for annotation in self.annotations if idx in annotation.channel_idxs]

    def active_channel_count(self) -> int:
        return int(np.count_nonzero(self.has_any_data))


@dataclass(slots=True)
class CaptionResult:
    rows: list[Recording] = field(default_factory=list)

    def iter_rows(self) -> Generator[Recording, None, None]:
        yield from self.rows

    def iter_channels(self) -> Generator[tuple[SignalView, list[Annotation]], None, None]:
        for row in self.rows:
            for signal in row.iter_channels():
                yield signal, row.annotations_for_signal(signal.idx)
