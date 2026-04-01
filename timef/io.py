#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from timef.schema import Annotation, DatasetManifest, Sample
from timef.utils import ensure_dir


@dataclass
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


@dataclass
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


@dataclass
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


def _serialize_channels(channels: list[str] | None) -> str | None:
    if channels is None:
        return None
    return json.dumps(channels)


class TimeFWriter:
    def __init__(self, root: Path):
        self.root = root
        self.signals_dir = ensure_dir(root / "signals")

    def write_manifest(self, manifest: DatasetManifest) -> None:
        ensure_dir(self.root)
        (self.root / "manifest.json").write_text(
            manifest.model_dump_json(indent=2, by_alias=True),
            encoding="utf-8",
        )

    def write_samples(self, rows: list[Sample]) -> None:
        records = [r.to_parquet_dict() for r in rows]
        df = pd.DataFrame(records)
        df.to_parquet(self.root / "samples.parquet", index=False)

        index_rows: list[SampleSignalIndexRow] = []
        for sample in rows:
            for sig_ref in sample.signals:
                index_rows.append(SampleSignalIndexRow(
                    sample_id=sample.id,
                    signal_id=sig_ref.signal_id,
                    sampling_rate=sig_ref.sampling_rate,
                    channels=_serialize_channels(sig_ref.channels),
                ))
        pd.DataFrame(
            [r.to_dict() for r in index_rows],
            columns=["sample_id", "signal_id", "sampling_rate", "channels"],
        ).to_parquet(self.root / "sample_signal_index.parquet", index=False)

    def write_annotations(self, rows: list[Annotation], manifest: DatasetManifest | None = None) -> None:
        records = [r.to_parquet_dict() for r in rows]
        df = pd.DataFrame(records)
        df.to_parquet(self.root / "annotations.parquet", index=False)

        task_lookup: dict[int, str] = {}
        domains: list[str] = []
        if manifest is not None:
            task_lookup = {spec.id: spec.task for spec in manifest.annotation_spec}
            domains = list(manifest.domains)

        task_rows: list[AnnotationTaskIndexRow] = []
        domain_rows: list[AnnotationDomainIndexRow] = []
        for ann in rows:
            task = task_lookup.get(ann.spec_id)
            for sample_ref in ann.samples:
                if task is not None:
                    task_rows.append(AnnotationTaskIndexRow(
                        sample_id=sample_ref.sample,
                        annotation_id=ann.id,
                        task_type=task,
                    ))
                for domain in domains:
                    domain_rows.append(AnnotationDomainIndexRow(
                        sample_id=sample_ref.sample,
                        annotation_id=ann.id,
                        domain=domain,
                    ))

        pd.DataFrame(
            [r.to_dict() for r in task_rows],
            columns=["sample_id", "annotation_id", "task_type"],
        ).to_parquet(self.root / "annotation_tasks_index.parquet", index=False)
        pd.DataFrame(
            [r.to_dict() for r in domain_rows],
            columns=["sample_id", "annotation_id", "domain"],
        ).to_parquet(self.root / "annotation_domains_index.parquet", index=False)

    def write_signal_frame(self, signal_file: str, frame: pd.DataFrame) -> None:
        frame.to_parquet(self.signals_dir / signal_file, index=False)


class TimeFReader:
    def __init__(self, root: Path):
        self.root = root

    def read_manifest(self) -> DatasetManifest:
        return DatasetManifest.model_validate(
            json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        )

    def read_samples(self, batch_size: int = 1024) -> Iterator[Sample]:
        parquet_file = pq.ParquetFile(self.root / "samples.parquet")
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            for row in df.itertuples(index=False):
                yield Sample.from_row(row)

    def read_samples_df(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "samples.parquet")

    def read_sample(self, sample_id: int) -> Sample | None:
        df = pd.read_parquet(self.root / "samples.parquet", filters=[("id", "==", sample_id)])
        if df.empty:
            return None
        row = next(iter(df.itertuples(index=False)))
        return Sample.from_row(row)

    def read_annotations(self, batch_size: int = 1024) -> Iterator[Annotation]:
        parquet_file = pq.ParquetFile(self.root / "annotations.parquet")
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            for row in df.itertuples(index=False):
                yield Annotation.from_row(row)

    def read_annotations_df(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "annotations.parquet")

    def _allowed_annotation_ids(
        self,
        annotation_tasks: list[str] | None = None,
        annotation_domains: list[str] | None = None,
    ) -> set[int] | None:
        tasks = [t for t in (annotation_tasks or []) if t]
        domains = [d for d in (annotation_domains or []) if d]
        allowed: set[int] | None = None

        if tasks:
            task_index = self.root / "annotation_tasks_index.parquet"
            if task_index.exists():
                task_df = pd.read_parquet(task_index, filters=[("task_type", "in", tasks)])
                if task_df.empty:
                    return set()
                counts = task_df.groupby("annotation_id")["task_type"].nunique()
                task_ids = set(counts[counts >= len(set(tasks))].index.tolist())
                allowed = task_ids

        if domains:
            domain_index = self.root / "annotation_domains_index.parquet"
            if domain_index.exists():
                domain_df = pd.read_parquet(domain_index, filters=[("domain", "in", domains)])
                if domain_df.empty:
                    return set()
                counts = domain_df.groupby("annotation_id")["domain"].nunique()
                domain_ids = set(counts[counts >= len(set(domains))].index.tolist())
                allowed = domain_ids if allowed is None else (allowed & domain_ids)

        return allowed

    def read_annotations_filtered(
        self,
        sample_id: int | None = None,
        task: str | None = None,
        domains: list[str] | None = None,
    ) -> Iterator[Annotation]:
        allowed_annotation_ids = self._allowed_annotation_ids(
            annotation_tasks=[task] if task else None,
            annotation_domains=domains,
        )

        filters = None
        if allowed_annotation_ids is not None:
            if not allowed_annotation_ids:
                return
            filters = [("id", "in", list(allowed_annotation_ids))]

        df = pd.read_parquet(self.root / "annotations.parquet", filters=filters)

        for row in df.itertuples(index=False):
            ann = Annotation.from_row(row)
            if sample_id is not None and not any(ref.sample == sample_id for ref in ann.samples):
                continue
            yield ann

    def query_samples(self, task: str | None = None, domains: list[str] | None = None) -> Iterator[Sample]:
        if not task and not domains:
            yield from self.read_samples()
            return

        allowed_annotation_ids = self._allowed_annotation_ids(
            annotation_tasks=[task] if task else None,
            annotation_domains=domains,
        )
        if allowed_annotation_ids is not None and not allowed_annotation_ids:
            return

        index_path = self.root / (
            "annotation_tasks_index.parquet" if task else "annotation_domains_index.parquet"
        )

        allowed_sample_ids: set[int] | None = None
        if index_path.exists() and allowed_annotation_ids is not None:
            index_df = pd.read_parquet(index_path, columns=["sample_id", "annotation_id"])
            index_df = index_df[index_df["annotation_id"].isin(allowed_annotation_ids)]
            allowed_sample_ids = set(index_df["sample_id"].tolist())
        else:
            allowed_sample_ids = set()
            for ann in self.read_annotations_filtered(task=task, domains=domains):
                for ref in ann.samples:
                    allowed_sample_ids.add(ref.sample)

        if not allowed_sample_ids:
            return

        df = pd.read_parquet(
            self.root / "samples.parquet",
            filters=[("id", "in", list(allowed_sample_ids))],
        )
        for row in df.itertuples(index=False):
            yield Sample.from_row(row)

    def read_signal_frame(self, signal_file: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / "signals" / signal_file)

    def read_signal_frame_for_sample(self, sample: Sample, manifest: DatasetManifest) -> dict[int, pd.DataFrame]:
        signals_by_id = {sig.id: sig for sig in manifest.signals}
        result: dict[int, pd.DataFrame] = {}
        for sig_ref in sample.signals:
            signal = signals_by_id.get(sig_ref.signal_id)
            if signal is None:
                continue
            parquet = pq.ParquetFile(self.root / "signals" / signal.shard_file)
            table = parquet.read_row_group(signal.row_group_id)
            df = table.to_pandas().copy()
            df = df.iloc[signal.row_start: signal.row_start + signal.row_count].copy()
            if sig_ref.channels is not None:
                keep_cols = [
                    c for c in df.columns
                    if c in sig_ref.channels or c == "time_s" or c == "sample_id"
                ]
                df = df[keep_cols]
            if "sample_id" in df.columns:
                df = df.drop(columns=["sample_id"])
            result[sig_ref.signal_id] = df.reset_index(drop=True)
        return result
