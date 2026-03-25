#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass(frozen=True)
class SignalSpec:
    id: str                          # e.g. "mhc:heart_rate"
    name: str                        # e.g. "heart_rate"
    display_name: str                # e.g. "Heart rate"
    channels: list[str]              # e.g. ["value"] (single-channel for MHC)
    unit_sampling_rate: str          # e.g. "1/min"
    unit_timestamp: str              # e.g. "minutes"
    unit_value: str                  # e.g. "bpm"
    sensor_id: str | None = None


@dataclass(frozen=True)
class Signal:
    id: str                          # e.g. "mhc:user42:2024-01-15:heart_rate"
    spec_id: str                     # -> SignalSpec.id
    name: str
    data: Any = field(repr=False)    # np.ndarray, in-memory for now
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SampleSignalRef:
    signal_id: str
    sampling_rate: float | None = None
    channels: list[str] | None = None   # None = all


@dataclass(frozen=True)
class Sample:
    id: str
    windows: list[tuple[int, int]] | None = None  # None = full signal
    signals: list[SampleSignalRef] = field(default_factory=list)


@dataclass(frozen=True)
class AnnotationSpec:
    id: str                          # e.g. "captioning:statistical"
    task: str                        # e.g. "captioning"
    schema: dict | None = None


@dataclass(frozen=True)
class SampleRef:
    sample_id: str
    reference: str | None = None     # alias for [ref:X] in text
    forecast: bool = False


@dataclass(frozen=True)
class Annotation:
    id: str
    spec_id: str                     # -> AnnotationSpec.id
    samples: list[SampleRef] = field(default_factory=list)
    question: str | None = None
    answer: str | None = None
    label: Any = None


@dataclass
class DatasetManifest:
    signal_specs: dict[str, SignalSpec] = field(default_factory=dict)
    annotation_specs: dict[str, AnnotationSpec] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class CaptionResult:
    manifest: DatasetManifest = field(default_factory=DatasetManifest)
    signals: dict[str, Signal] = field(default_factory=dict)
    samples: list[Sample] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)

    def iter_rows(self) -> Generator[tuple[list[Signal], list[Sample], list[Annotation]], None, None]:
        rows: dict[object, list[Signal]] = {}
        for signal in self.signals.values():
            row_id = signal.metadata.get("row_id")
            key = row_id if row_id is not None else tuple(sorted(signal.metadata.items()))
            rows.setdefault(key, []).append(signal)

        sample_to_annotations: dict[str, list[Annotation]] = {}
        for ann in self.annotations:
            for sr in ann.samples:
                sample_to_annotations.setdefault(sr.sample_id, []).append(ann)

        for signals in rows.values():
            signal_ids = {s.id for s in signals}

            # Find samples referencing this row's signals
            row_samples = [
                s for s in self.samples
                if any(ref.signal_id in signal_ids for ref in s.signals)
            ]

            # Find annotations referencing this row's samples
            row_annotations = []
            seen = set()
            for sample in row_samples:
                for ann in sample_to_annotations.get(sample.id, []):
                    if ann.id not in seen:
                        seen.add(ann.id)
                        row_annotations.append(ann)

            yield signals, row_samples, row_annotations
