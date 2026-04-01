#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from timef.io import TimeFReader
from timef.schema import VALID_DOMAINS, validate_annotation_against_spec


class TimeFValidationError(ValueError):
    pass


def validate_dataset(root: Path) -> None:
    reader = TimeFReader(root)
    manifest = reader.read_manifest()
    samples = list(reader.read_samples())
    annotations = list(reader.read_annotations())

    if manifest.sample_count != len(samples):
        raise TimeFValidationError("manifest.sample_count does not match samples.parquet")
    if manifest.annotation_count != len(annotations):
        raise TimeFValidationError("manifest.annotation_count does not match annotations.parquet")

    invalid_domains = set(manifest.domains) - VALID_DOMAINS
    if invalid_domains:
        raise TimeFValidationError(f"invalid manifest domains: {invalid_domains}")

    signal_spec_by_id = {s.id: s for s in manifest.signal_spec}
    sensor_spec_ids = {s.id for s in manifest.sensor_spec}
    annotation_spec_by_id = {s.id: s for s in manifest.annotation_spec}
    signal_by_id = {sig.id: sig for sig in manifest.signals}
    sample_ids = {s.id for s in samples}

    for sig in manifest.signals:
        if sig.spec_id not in signal_spec_by_id:
            raise TimeFValidationError(f"signal {sig.id} references unknown spec_id {sig.spec_id}")

    for spec in manifest.signal_spec:
        if spec.sensor_id is not None and spec.sensor_id not in sensor_spec_ids:
            raise TimeFValidationError(
                f"signal_spec {spec.id} references unknown sensor_id {spec.sensor_id}"
            )

    for aspec in manifest.annotation_spec:
        invalid = set(aspec.domains) - VALID_DOMAINS
        if invalid:
            raise TimeFValidationError(f"annotation_spec {aspec.id} has invalid domains: {invalid}")

    for sig in manifest.signals:
        shard_path = root / "signals" / sig.shard_file
        if not shard_path.exists():
            raise TimeFValidationError(f"missing shard file {sig.shard_file}")
        meta = pq.ParquetFile(shard_path).metadata
        if sig.row_group_id >= meta.num_row_groups:
            raise TimeFValidationError(
                f"row_group_id out of bounds for signal {sig.id} in {sig.shard_file}"
            )
        rg_rows = meta.row_group(sig.row_group_id).num_rows
        if sig.row_start + sig.row_count > rg_rows:
            raise TimeFValidationError(
                f"row span out of bounds for signal {sig.id} in row group {sig.row_group_id}"
            )

    if not (root / "sample_signal_index.parquet").exists():
        raise TimeFValidationError("missing sample_signal_index.parquet")
    if not (root / "annotation_tasks_index.parquet").exists():
        raise TimeFValidationError("missing annotation_tasks_index.parquet")
    if not (root / "annotation_domains_index.parquet").exists():
        raise TimeFValidationError("missing annotation_domains_index.parquet")

    for sample in samples:
        for sig_ref in sample.signals:
            if sig_ref.signal_id not in signal_by_id:
                raise TimeFValidationError(
                    f"sample {sample.id} references unknown signal_id {sig_ref.signal_id}"
                )
            if sig_ref.channels is not None:
                signal = signal_by_id[sig_ref.signal_id]
                spec = signal_spec_by_id.get(signal.spec_id)
                if spec is not None and spec.channels:
                    invalid_channels = set(sig_ref.channels) - set(spec.channels)
                    if invalid_channels:
                        raise TimeFValidationError(
                            f"sample {sample.id} selects channels {invalid_channels} "
                            f"not present in signal spec {spec.id}"
                        )

    for ann in annotations:
        if ann.spec_id not in annotation_spec_by_id:
            raise TimeFValidationError(
                f"annotation {ann.id} references unknown spec_id {ann.spec_id}"
            )
        for ref in ann.samples:
            if ref.sample not in sample_ids:
                raise TimeFValidationError(f"annotation {ann.id} references unknown sample {ref.sample}")
        has_answer = ann.answer is not None
        has_label = ann.label is not None
        if has_answer == has_label:
            raise TimeFValidationError(
                f"annotation {ann.id} must have exactly one of answer or label"
            )
        spec = annotation_spec_by_id[ann.spec_id]
        validate_annotation_against_spec(ann, spec)


def mark_validated(root: Path, validated: bool = True) -> None:
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["validated"] = validated
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
