#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from timef.labels import get_valid_labels
from timef.units import ucum_to_pint, validate_ucum

if TYPE_CHECKING:
    from pint import Unit


class SampleRow(NamedTuple):
    """NamedTuple matching the samples.parquet schema for efficient iteration."""

    id: int
    dataset_id: str
    unit_timestamp: str
    windows: Optional[str]
    signals: str


class AnnotationRow(NamedTuple):
    """NamedTuple matching the annotations.parquet schema for efficient iteration."""

    id: int
    dataset_id: str
    spec_id: int
    samples: str
    question: Optional[str]
    answer: Optional[str]
    label: Optional[str]
    rationale: Optional[str]


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


class SensorSpec(BaseModel):
    id: int = Field(ge=0)
    name: str


class NormalizedSampleRow(BaseModel):
    sample_id: str
    duration_s: float = Field(gt=0)
    signal_file: str
    signal_ids: list[str] = Field(min_length=1)


class NormalizedAnnotationRow(BaseModel):
    sample_id: str
    annotation_id: str
    type: Literal["span", "global"] = "global"
    question: str
    answer: str
    task_types: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    sensors: Optional[list[str]] = None
    window_s: Optional[list[float]] = None

    @model_validator(mode="after")
    def validate_semantics(self) -> NormalizedAnnotationRow:
        if self.type == "span":
            if not self.sensors:
                raise ValueError("span annotations require sensors")
            if not self.window_s or len(self.window_s) != 2:
                raise ValueError("span annotations require window_s=[start,end]")
            if self.window_s[0] >= self.window_s[1]:
                raise ValueError("span annotation window must satisfy start < end")
        return self


class SignalSpec(BaseModel):
    id: int = Field(ge=0)
    name: str
    channels: list[str] = Field(default_factory=list)
    unit_sampling_rate: str
    unit_timestamp: str
    unit_value: Optional[str] = None
    sensor_id: Optional[int] = None

    @field_validator("unit_sampling_rate", "unit_timestamp")
    @classmethod
    def check_ucum_required(cls, v: str) -> str:
        validate_ucum(v)
        return v

    @field_validator("unit_value")
    @classmethod
    def check_ucum_optional(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            validate_ucum(v)
        return v

    def get_pint_unit(self, field: str = "unit_value") -> Optional[Unit]:
        val = getattr(self, field, None)
        if val is None:
            return None
        return ucum_to_pint(val)


class Signal(BaseModel):
    id: int = Field(ge=0)
    spec_id: int = Field(ge=0)
    name: str
    sampling_rate: Optional[float] = Field(default=None, gt=0)
    total_duration: float = Field(ge=0)
    shard_file: str
    row_group_id: int = Field(ge=0)
    row_start: int = Field(ge=0, default=0)
    row_count: int = Field(gt=0)


class SampleSignalRef(BaseModel):
    signal_id: int = Field(ge=0)
    sampling_rate: Optional[float] = Field(default=None, gt=0)
    channels: Optional[list[str]] = None


class Sample(BaseModel):
    id: int = Field(ge=0)
    dataset_id: str
    unit_timestamp: str
    windows: Optional[list[list[float]]] = None
    signals: list[SampleSignalRef] = Field(min_length=1)

    @field_validator("unit_timestamp")
    @classmethod
    def check_ucum_timestamp(cls, v: str) -> str:
        validate_ucum(v)
        return v

    @model_validator(mode="after")
    def validate_windows(self) -> "Sample":
        if self.windows is not None:
            for w in self.windows:
                if len(w) != 2:
                    raise ValueError("each window must be [start, end]")
                if w[0] >= w[1]:
                    raise ValueError("window start must be < end")
        return self

    def to_parquet_dict(self) -> dict:
        d = self.model_dump()
        d["windows"] = json.dumps(d["windows"])
        d["signals"] = json.dumps(d["signals"])
        return d

    @classmethod
    def from_parquet_dict(cls, d: dict) -> "Sample":
        d = dict(d)
        if isinstance(d.get("windows"), str):
            d["windows"] = json.loads(d["windows"])
        if isinstance(d.get("signals"), str):
            d["signals"] = json.loads(d["signals"])
        return cls.model_validate(d)

    @classmethod
    def from_row(cls, row: Any) -> "Sample":
        windows = json.loads(row.windows) if isinstance(row.windows, str) else row.windows
        signals = json.loads(row.signals) if isinstance(row.signals, str) else row.signals
        return cls.model_validate({
            "id": row.id,
            "dataset_id": row.dataset_id,
            "unit_timestamp": row.unit_timestamp,
            "windows": windows,
            "signals": signals,
        })


class AnnotationSpec(BaseModel):
    id: int = Field(ge=0)
    task: str
    schema_: Optional[str] = Field(default=None, alias="schema")
    domains: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def check_label_source(self) -> "AnnotationSpec":
        if self.task in {"labeling", "classification"}:
            if not self.schema_:
                raise ValueError("Labeling/classification tasks require a schema")
            get_valid_labels(self.schema_)
        return self

    def get_valid_labels(self) -> Optional[list[str]]:
        if self.schema_:
            return get_valid_labels(self.schema_)
        return None


class AnnotationSampleRef(BaseModel):
    sample: int = Field(ge=0)
    reference: Optional[str] = None
    forecast: bool = False


class Annotation(BaseModel):
    id: int = Field(ge=0)
    dataset_id: str
    spec_id: int = Field(ge=0)
    samples: list[AnnotationSampleRef] = Field(min_length=1)
    question: Optional[str] = None
    answer: Optional[str] = None
    label: Optional[str] = None
    rationale: Optional[str] = None

    @model_validator(mode="after")
    def validate_answer_xor_label(self) -> "Annotation":
        has_answer = self.answer is not None
        has_label = self.label is not None
        if has_answer == has_label:
            raise ValueError("exactly one of 'answer' or 'label' must be provided")
        return self

    def get_formatted_answer(self) -> Optional[str]:
        if self.rationale and self.label:
            return f"{self.rationale} Answer: {self.label}"
        if self.rationale and self.answer:
            return f"{self.rationale} Answer: {self.answer}"
        if self.label:
            return f"Answer: {self.label}"
        return self.answer

    def to_parquet_dict(self) -> dict:
        d = self.model_dump()
        d["samples"] = json.dumps(d["samples"])
        return d

    @classmethod
    def from_parquet_dict(cls, d: dict) -> "Annotation":
        d = dict(d)
        if isinstance(d.get("samples"), str):
            d["samples"] = json.loads(d["samples"])
        return cls.model_validate(d)

    @classmethod
    def from_row(cls, row: Any) -> "Annotation":
        samples = row.samples
        if isinstance(samples, str):
            samples = json.loads(samples)
        return cls.model_validate({
            "id": row.id,
            "dataset_id": row.dataset_id,
            "spec_id": row.spec_id,
            "samples": samples,
            "question": getattr(row, "question", None),
            "answer": getattr(row, "answer", None),
            "label": getattr(row, "label", None),
            "rationale": getattr(row, "rationale", None),
        })


class DatasetManifest(BaseModel):
    dataset_id: str
    version: str
    source: str
    description: str = ""
    domains: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    signals: list[Signal] = Field(default_factory=list)
    signal_spec: list[SignalSpec] = Field(default_factory=list)
    sensor_spec: list[SensorSpec] = Field(default_factory=list)
    annotation_spec: list[AnnotationSpec] = Field(default_factory=list)
    sample_count: int = Field(ge=0)
    annotation_count: int = Field(ge=0)
    validated: bool = False

    @model_validator(mode="after")
    def validate_domains(self) -> "DatasetManifest":
        invalid = set(self.domains) - VALID_DOMAINS
        if invalid:
            raise ValueError(f"invalid domains: {invalid}. Valid: {sorted(VALID_DOMAINS)}")
        return self


def validate_annotation_against_spec(annotation: Annotation, spec: AnnotationSpec) -> None:
    if spec.task in {"labeling", "classification"}:
        if annotation.label is None:
            raise ValueError(f"Classification annotation '{annotation.id}' requires a label")
        valid_labels = spec.get_valid_labels()
        if valid_labels and annotation.label not in valid_labels:
            raise ValueError(
                f"Invalid label '{annotation.label}' for annotation spec '{spec.id}'. "
                f"Valid labels: {valid_labels}"
            )
