#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from timenet_timef.labels import get_valid_labels
from timenet_timef.units import validate_ucum


VALID_DOMAINS: frozenset[str] = frozenset(
    {
        "health",
        "activity",
        "general",
        "forecasting",
        "economics",
        "sleep",
        "cardiology",
        "finance",
        "energy",
        "environment",
        "industrial",
        "transport",
    }
)


@dataclass(slots=True)
class SensorSpec:
    id: int
    name: str


@dataclass(slots=True)
class SignalSpec:
    id: int
    name: str
    channels: list[str] = field(default_factory=list)
    unit_sampling_rate: str = "Hz"
    unit_timestamp: str = "s"
    unit_value: str | None = None
    sensor_id: int | None = None

    def __post_init__(self) -> None:
        validate_ucum(self.unit_sampling_rate)
        validate_ucum(self.unit_timestamp)
        if self.unit_value is not None:
            validate_ucum(self.unit_value)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SignalSpec":
        return cls(**payload)


@dataclass(slots=True)
class Signal:
    id: int
    spec_id: int
    name: str
    sampling_rate: float | None
    total_duration: float
    shard_file: str
    row_group_id: int
    row_start: int = 0
    row_count: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Signal":
        return cls(**payload)


@dataclass(slots=True)
class SampleSignalRef:
    signal_id: int
    sampling_rate: float | None = None
    channels: list[str] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SampleSignalRef":
        return cls(**payload)


@dataclass(slots=True)
class Sample:
    id: int
    dataset_id: str
    unit_timestamp: str
    windows: list[list[float]] | None = None
    signals: list[SampleSignalRef] = field(default_factory=list)

    def __post_init__(self) -> None:
        validate_ucum(self.unit_timestamp)
        if not self.signals:
            raise ValueError("Sample must reference at least one signal")
        if self.windows is not None:
            for window in self.windows:
                if len(window) != 2 or window[0] >= window[1]:
                    raise ValueError("Sample windows must be [start, end] with start < end")

    def to_parquet_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "unit_timestamp": self.unit_timestamp,
            "windows": json.dumps(self.windows),
            "signals": json.dumps([asdict(sig) for sig in self.signals]),
        }

    @classmethod
    def from_parquet_dict(cls, payload: dict[str, Any]) -> "Sample":
        windows = payload.get("windows")
        signals = payload.get("signals")
        if isinstance(windows, str):
            windows = json.loads(windows)
        if isinstance(signals, str):
            signals = json.loads(signals)
        return cls(
            id=int(payload["id"]),
            dataset_id=str(payload["dataset_id"]),
            unit_timestamp=str(payload["unit_timestamp"]),
            windows=windows,
            signals=[SampleSignalRef.from_dict(sig) for sig in signals],
        )


@dataclass(slots=True)
class AnnotationSpec:
    id: int
    task: str
    schema_: str | None = None
    domains: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        invalid = set(self.domains) - VALID_DOMAINS
        if invalid:
            raise ValueError(f"Invalid annotation domains: {sorted(invalid)}")
        if self.task in {"labeling", "classification"} and self.schema_ is None:
            raise ValueError("Classification annotation specs require a schema")
        if self.schema_ is not None:
            get_valid_labels(self.schema_)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "schema": self.schema_,
            "domains": self.domains,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnnotationSpec":
        data = dict(payload)
        if "schema" in data and "schema_" not in data:
            data["schema_"] = data.pop("schema")
        return cls(**data)

    def get_valid_labels(self) -> list[str] | None:
        if self.schema_ is None:
            return None
        return get_valid_labels(self.schema_)


@dataclass(slots=True)
class AnnotationSampleRef:
    sample: int
    reference: str | None = None
    forecast: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnnotationSampleRef":
        return cls(**payload)


@dataclass(slots=True)
class Annotation:
    id: int
    dataset_id: str
    spec_id: int
    samples: list[AnnotationSampleRef]
    question: str | None = None
    answer: str | None = None
    label: str | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("Annotation must reference at least one sample")
        has_answer = self.answer is not None
        has_label = self.label is not None
        if has_answer == has_label:
            raise ValueError("Annotation must have exactly one of answer or label")

    def to_parquet_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "spec_id": self.spec_id,
            "samples": json.dumps([asdict(sample) for sample in self.samples]),
            "question": self.question,
            "answer": self.answer,
            "label": self.label,
            "rationale": self.rationale,
        }

    @classmethod
    def from_parquet_dict(cls, payload: dict[str, Any]) -> "Annotation":
        samples = payload.get("samples")
        if isinstance(samples, str):
            samples = json.loads(samples)
        return cls(
            id=int(payload["id"]),
            dataset_id=str(payload["dataset_id"]),
            spec_id=int(payload["spec_id"]),
            samples=[AnnotationSampleRef.from_dict(sample) for sample in samples],
            question=payload.get("question"),
            answer=payload.get("answer"),
            label=payload.get("label"),
            rationale=payload.get("rationale"),
        )


@dataclass(slots=True)
class DatasetManifest:
    dataset_id: str
    version: str
    source: str
    description: str = ""
    domains: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    signal_spec: list[SignalSpec] = field(default_factory=list)
    sensor_spec: list[SensorSpec] = field(default_factory=list)
    annotation_spec: list[AnnotationSpec] = field(default_factory=list)
    sample_count: int = 0
    annotation_count: int = 0
    validated: bool = False

    def __post_init__(self) -> None:
        invalid = set(self.domains) - VALID_DOMAINS
        if invalid:
            raise ValueError(f"Invalid manifest domains: {sorted(invalid)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "source": self.source,
            "description": self.description,
            "domains": self.domains,
            "tags": self.tags,
            "signals": [asdict(signal) for signal in self.signals],
            "signal_spec": [asdict(spec) for spec in self.signal_spec],
            "sensor_spec": [asdict(spec) for spec in self.sensor_spec],
            "annotation_spec": [spec.to_dict() for spec in self.annotation_spec],
            "sample_count": self.sample_count,
            "annotation_count": self.annotation_count,
            "validated": self.validated,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetManifest":
        return cls(
            dataset_id=payload["dataset_id"],
            version=payload["version"],
            source=payload["source"],
            description=payload.get("description", ""),
            domains=list(payload.get("domains", [])),
            tags=list(payload.get("tags", [])),
            signals=[Signal.from_dict(item) for item in payload.get("signals", [])],
            signal_spec=[SignalSpec.from_dict(item) for item in payload.get("signal_spec", [])],
            sensor_spec=[SensorSpec(**item) for item in payload.get("sensor_spec", [])],
            annotation_spec=[AnnotationSpec.from_dict(item) for item in payload.get("annotation_spec", [])],
            sample_count=int(payload.get("sample_count", 0)),
            annotation_count=int(payload.get("annotation_count", 0)),
            validated=bool(payload.get("validated", False)),
        )


def validate_annotation_against_spec(annotation: Annotation, spec: AnnotationSpec) -> None:
    if spec.task not in {"labeling", "classification"}:
        return
    if annotation.label is None:
        raise ValueError(f"Classification annotation {annotation.id} requires a label")
    valid_labels = spec.get_valid_labels()
    if valid_labels is not None and annotation.label not in valid_labels:
        raise ValueError(
            f"Invalid label '{annotation.label}' for annotation spec {spec.id}. Valid labels: {valid_labels}"
        )
