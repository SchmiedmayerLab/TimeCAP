#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from timenet_timef.schema import Annotation, DatasetManifest, Sample
from timenet_timef.utils import ensure_dir


def _table_from_rows(rows: list[dict[str, Any]]) -> pa.Table:
    if not rows:
        return pa.table({})
    return pa.Table.from_pylist(rows)


@dataclass(slots=True)
class SampleSignalIndexRow:
    sample_id: int
    signal_id: int
    sampling_rate: float | None
    channels: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "signal_id": self.signal_id,
            "sampling_rate": self.sampling_rate,
            "channels": self.channels,
        }


@dataclass(slots=True)
class AnnotationTaskIndexRow:
    sample_id: int
    annotation_id: int
    task_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "annotation_id": self.annotation_id,
            "task_type": self.task_type,
        }


@dataclass(slots=True)
class AnnotationDomainIndexRow:
    sample_id: int
    annotation_id: int
    domain: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "annotation_id": self.annotation_id,
            "domain": self.domain,
        }


class TimeFWriter:
    def __init__(self, root: Path):
        self.root = root
        self.signals_dir = ensure_dir(root / "signals")

    def write_manifest(self, manifest: DatasetManifest) -> None:
        ensure_dir(self.root)
        (self.root / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2),
            encoding="utf-8",
        )

    def write_samples(self, rows: list[Sample]) -> None:
        pq.write_table(_table_from_rows([row.to_parquet_dict() for row in rows]), self.root / "samples.parquet")

        index_rows: list[dict[str, Any]] = []
        for sample in rows:
            for signal_ref in sample.signals:
                channels = None if signal_ref.channels is None else json.dumps(signal_ref.channels)
                index_rows.append(
                    SampleSignalIndexRow(
                        sample_id=sample.id,
                        signal_id=signal_ref.signal_id,
                        sampling_rate=signal_ref.sampling_rate,
                        channels=channels,
                    ).to_dict()
                )
        pq.write_table(_table_from_rows(index_rows), self.root / "sample_signal_index.parquet")

    def write_annotations(self, rows: list[Annotation], manifest: DatasetManifest | None = None) -> None:
        annotation_rows = [row.to_parquet_dict() for row in rows]
        if annotation_rows:
            annotation_table = _table_from_rows(annotation_rows)
        else:
            annotation_table = pa.table(
                {
                    "id": pa.array([], type=pa.int64()),
                    "dataset_id": pa.array([], type=pa.string()),
                    "spec_id": pa.array([], type=pa.int64()),
                    "samples": pa.array([], type=pa.string()),
                    "question": pa.array([], type=pa.string()),
                    "answer": pa.array([], type=pa.string()),
                    "label": pa.array([], type=pa.string()),
                    "rationale": pa.array([], type=pa.string()),
                }
            )
        pq.write_table(annotation_table, self.root / "annotations.parquet")

        task_lookup: dict[int, str] = {}
        domains: list[str] = []
        if manifest is not None:
            task_lookup = {spec.id: spec.task for spec in manifest.annotation_spec}
            domains = list(manifest.domains)

        task_rows: list[dict[str, Any]] = []
        domain_rows: list[dict[str, Any]] = []
        for annotation in rows:
            task_type = task_lookup.get(annotation.spec_id)
            for sample_ref in annotation.samples:
                if task_type is not None:
                    task_rows.append(
                        AnnotationTaskIndexRow(
                            sample_id=sample_ref.sample,
                            annotation_id=annotation.id,
                            task_type=task_type,
                        ).to_dict()
                    )
                for domain in domains:
                    domain_rows.append(
                        AnnotationDomainIndexRow(
                            sample_id=sample_ref.sample,
                            annotation_id=annotation.id,
                            domain=domain,
                        ).to_dict()
                    )

        if task_rows:
            task_table = _table_from_rows(task_rows)
        else:
            task_table = pa.table(
                {
                    "sample_id": pa.array([], type=pa.int64()),
                    "annotation_id": pa.array([], type=pa.int64()),
                    "task_type": pa.array([], type=pa.string()),
                }
            )
        if domain_rows:
            domain_table = _table_from_rows(domain_rows)
        else:
            domain_table = pa.table(
                {
                    "sample_id": pa.array([], type=pa.int64()),
                    "annotation_id": pa.array([], type=pa.int64()),
                    "domain": pa.array([], type=pa.string()),
                }
            )

        pq.write_table(task_table, self.root / "annotation_tasks_index.parquet")
        pq.write_table(domain_table, self.root / "annotation_domains_index.parquet")

    def write_signal_frame(self, signal_file: str, frame: dict[str, list[Any]]) -> None:
        pq.write_table(pa.table(frame), self.signals_dir / signal_file)


class TimeFReader:
    def __init__(self, root: Path):
        self.root = root

    def read_manifest(self) -> DatasetManifest:
        payload = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        return DatasetManifest.from_dict(payload)

    def read_samples(self) -> Iterator[Sample]:
        table = pq.read_table(self.root / "samples.parquet")
        for row in table.to_pylist():
            yield Sample.from_parquet_dict(row)

    def read_sample(self, sample_id: int) -> Sample | None:
        for sample in self.read_samples():
            if sample.id == sample_id:
                return sample
        return None

    def read_annotations(self) -> Iterator[Annotation]:
        table = pq.read_table(self.root / "annotations.parquet")
        for row in table.to_pylist():
            yield Annotation.from_parquet_dict(row)

    def read_signal_frame(self, signal_file: str) -> list[dict[str, Any]]:
        return pq.read_table(self.root / "signals" / signal_file).to_pylist()

    def read_signal_frame_for_sample(self, sample: Sample, manifest: DatasetManifest) -> dict[int, list[dict[str, Any]]]:
        signals_by_id = {signal.id: signal for signal in manifest.signals}
        result: dict[int, list[dict[str, Any]]] = {}
        for signal_ref in sample.signals:
            signal = signals_by_id.get(signal_ref.signal_id)
            if signal is None:
                continue
            rows = self.read_signal_frame(signal.shard_file)
            sliced = rows[signal.row_start : signal.row_start + signal.row_count]
            if signal_ref.channels is not None:
                filtered: list[dict[str, Any]] = []
                for row in sliced:
                    filtered.append(
                        {
                            key: value
                            for key, value in row.items()
                            if key == "time_s" or key in signal_ref.channels
                        }
                    )
                sliced = filtered
            result[signal_ref.signal_id] = sliced
        return result
