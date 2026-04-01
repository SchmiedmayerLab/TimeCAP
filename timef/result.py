#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generator

from timef.schema import AnnotationSpec, SignalSpec


@dataclass(frozen=True)
class Signal:
    id: str
    spec_id: int
    name: str
    data: Any = field(repr=False)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SampleSignalRef:
    signal_id: str
    sampling_rate: float | None = None
    channels: list[str] | None = None


@dataclass(frozen=True)
class Sample:
    id: str
    windows: list[tuple[int, int]] | None = None
    signals: list[SampleSignalRef] = field(default_factory=list)


@dataclass(frozen=True)
class SampleRef:
    sample_id: str
    reference: str | None = None
    forecast: bool = False


@dataclass(frozen=True)
class Annotation:
    id: str
    spec_id: int
    samples: list[SampleRef] = field(default_factory=list)
    question: str | None = None
    answer: str | None = None
    label: Any = None


@dataclass
class RuntimeManifest:
    signal_specs: list[SignalSpec] = field(default_factory=list)
    annotation_specs: list[AnnotationSpec] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class CaptionResult:
    manifest: RuntimeManifest = field(default_factory=RuntimeManifest)
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
            row_samples = [
                sample for sample in self.samples
                if any(ref.signal_id in signal_ids for ref in sample.signals)
            ]

            row_annotations = []
            seen = set()
            for sample in row_samples:
                for ann in sample_to_annotations.get(sample.id, []):
                    if ann.id not in seen:
                        seen.add(ann.id)
                        row_annotations.append(ann)

            yield signals, row_samples, row_annotations
