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

from timenet_timef.schema import VALID_DOMAINS, validate_annotation_against_spec
from timenet_timef.io import TimeFReader


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
        raise TimeFValidationError(f"invalid manifest domains: {sorted(invalid_domains)}")

    signal_spec_by_id = {spec.id: spec for spec in manifest.signal_spec}
    signal_by_id = {signal.id: signal for signal in manifest.signals}
    annotation_spec_by_id = {spec.id: spec for spec in manifest.annotation_spec}
    sample_ids = {sample.id for sample in samples}

    for required in (
        "manifest.json",
        "samples.parquet",
        "annotations.parquet",
        "sample_signal_index.parquet",
        "annotation_tasks_index.parquet",
        "annotation_domains_index.parquet",
    ):
        if not (root / required).exists():
            raise TimeFValidationError(f"missing required file {required}")

    for signal in manifest.signals:
        if signal.spec_id not in signal_spec_by_id:
            raise TimeFValidationError(f"signal {signal.id} references unknown spec_id {signal.spec_id}")
        shard_path = root / "signals" / signal.shard_file
        if not shard_path.exists():
            raise TimeFValidationError(f"missing shard file {signal.shard_file}")
        metadata = pq.ParquetFile(shard_path).metadata
        if signal.row_group_id >= metadata.num_row_groups:
            raise TimeFValidationError(f"row_group_id out of bounds for signal {signal.id}")
        row_group_rows = metadata.row_group(signal.row_group_id).num_rows
        if signal.row_start + signal.row_count > row_group_rows:
            raise TimeFValidationError(f"row span out of bounds for signal {signal.id}")

    for sample in samples:
        for signal_ref in sample.signals:
            if signal_ref.signal_id not in signal_by_id:
                raise TimeFValidationError(
                    f"sample {sample.id} references unknown signal_id {signal_ref.signal_id}"
                )
            if signal_ref.channels is not None:
                spec = signal_spec_by_id[signal_by_id[signal_ref.signal_id].spec_id]
                invalid_channels = set(signal_ref.channels) - set(spec.channels)
                if invalid_channels:
                    raise TimeFValidationError(
                        f"sample {sample.id} selects invalid channels {sorted(invalid_channels)}"
                    )

    for annotation in annotations:
        spec = annotation_spec_by_id.get(annotation.spec_id)
        if spec is None:
            raise TimeFValidationError(f"annotation {annotation.id} references unknown spec_id {annotation.spec_id}")
        for ref in annotation.samples:
            if ref.sample not in sample_ids:
                raise TimeFValidationError(
                    f"annotation {annotation.id} references unknown sample {ref.sample}"
                )
            if ref.reference is not None:
                try:
                    json.loads(ref.reference)
                except json.JSONDecodeError as exc:
                    raise TimeFValidationError(
                        f"annotation {annotation.id} has invalid reference JSON"
                    ) from exc
        try:
            validate_annotation_against_spec(annotation, spec)
        except ValueError as exc:
            raise TimeFValidationError(str(exc)) from exc


def mark_validated(root: Path, validated: bool = True) -> None:
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["validated"] = validated
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
