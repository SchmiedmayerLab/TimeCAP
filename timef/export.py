from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from timef.schema import CaptionResult
from timenet_timef import (
    Annotation as PersistedAnnotation,
    AnnotationSampleRef,
    AnnotationSpec,
    DatasetManifest,
    Sample,
    SampleSignalRef,
    Signal,
    SignalSpec,
    TimeFWriter,
    mark_validated,
    validate_dataset,
)

_SPEC_IDS = {
    "statistical": 0,
    "structural": 1,
    "semantic": 2,
}

_QUESTIONS = {
    "statistical": "Describe summary statistics for the selected signal.",
    "structural": "Describe the structural event in the selected signal.",
    "semantic": "Describe the semantic event in the selected signal.",
}


@dataclass(frozen=True)
class TimeFExportConfig:
    output_root: Path
    dataset_id: str
    version: str | None = None
    source: str = "sensortslm"
    description: str = ""
    domains: tuple[str, ...] = ("health", "activity", "sleep")
    tags: tuple[str, ...] = ("sensortslm", "generated")
    sampling_period_seconds: float = 60.0
    value_unit_map: dict[str, str] | None = None


def export_caption_result(result: CaptionResult, config: TimeFExportConfig) -> Path:
    version = config.version or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path(config.output_root) / config.dataset_id / version
    if root.exists():
        raise FileExistsError(f"Export target already exists: {root}")
    if not result.rows:
        raise ValueError("CaptionResult is empty")
    if config.sampling_period_seconds <= 0:
        raise ValueError("sampling_period_seconds must be > 0")

    first_row = result.rows[0]
    channel_names = tuple(first_row.channel_names)
    signal_specs = _build_signal_specs(channel_names, config)
    annotation_specs = [
        AnnotationSpec(id=_SPEC_IDS["statistical"], task="statistical", domains=list(config.domains)),
        AnnotationSpec(id=_SPEC_IDS["structural"], task="structural", domains=list(config.domains)),
        AnnotationSpec(id=_SPEC_IDS["semantic"], task="semantic", domains=list(config.domains)),
    ]

    writer = TimeFWriter(root)
    samples: list[Sample] = []
    signals: list[Signal] = []
    annotations: list[PersistedAnnotation] = []
    sampling_rate = 1.0 / config.sampling_period_seconds
    annotation_id = 0
    for sample_id, row in enumerate(result.rows):
        _validate_row_shape(row, channel_names)
        signal_file = f"sample-{sample_id}.parquet"
        _write_signal_frame(root / "signals" / signal_file, row, config.sampling_period_seconds)
        total_duration = float(row.values.shape[1] * config.sampling_period_seconds)

        signal_refs: list[SampleSignalRef] = []
        for channel_idx, channel_name in enumerate(channel_names):
            signal_id = sample_id * len(channel_names) + channel_idx
            signals.append(
                Signal(
                    id=signal_id,
                    spec_id=channel_idx,
                    name=f"{row.row_id}:{channel_name}",
                    sampling_rate=sampling_rate,
                    total_duration=total_duration,
                    shard_file=signal_file,
                    row_group_id=0,
                    row_start=0,
                    row_count=row.values.shape[1],
                )
            )
            signal_refs.append(
                SampleSignalRef(
                    signal_id=signal_id,
                    sampling_rate=sampling_rate,
                    channels=[channel_name],
                )
            )

        samples.append(
            Sample(
                id=sample_id,
                dataset_id=config.dataset_id,
                unit_timestamp="s",
                windows=None,
                signals=signal_refs,
            )
        )

        for annotation in row.annotations:
            spec_id = _SPEC_IDS.get(annotation.kind)
            if spec_id is None:
                raise ValueError(f"Unsupported annotation kind: {annotation.kind}")
            reference = json.dumps(
                {
                    "channel_names": [channel_names[idx] for idx in annotation.channel_idxs],
                    "window": list(annotation.window) if annotation.window is not None else None,
                    "kind": annotation.kind,
                },
                sort_keys=True,
            )

            label = None if annotation.label is None else str(annotation.label)
            answer = annotation.text if label is None else None
            rationale = annotation.text if label is not None and annotation.text else None

            annotations.append(
                PersistedAnnotation(
                    id=annotation_id,
                    dataset_id=config.dataset_id,
                    spec_id=spec_id,
                    samples=[AnnotationSampleRef(sample=sample_id, reference=reference)],
                    question=_QUESTIONS[annotation.kind],
                    answer=answer,
                    label=label,
                    rationale=rationale,
                )
            )
            annotation_id += 1

    manifest = DatasetManifest(
        dataset_id=config.dataset_id,
        version=version,
        source=config.source,
        description=config.description,
        domains=list(config.domains),
        tags=list(config.tags),
        signals=signals,
        signal_spec=signal_specs,
        sensor_spec=[],
        annotation_spec=annotation_specs,
        sample_count=len(samples),
        annotation_count=len(annotations),
        validated=False,
    )

    writer.write_manifest(manifest)
    writer.write_samples(samples)
    writer.write_annotations(annotations, manifest)
    validate_dataset(root)
    mark_validated(root, validated=True)
    return root


def _build_signal_specs(channel_names: tuple[str, ...], config: TimeFExportConfig) -> list[SignalSpec]:
    value_unit_map = config.value_unit_map or {}
    return [
        SignalSpec(
            id=idx,
            name=channel_name,
            channels=[channel_name],
            unit_sampling_rate="Hz",
            unit_timestamp="s",
            unit_value=value_unit_map.get(channel_name),
        )
        for idx, channel_name in enumerate(channel_names)
    ]


def _validate_row_shape(row, channel_names: tuple[str, ...]) -> None:
    if tuple(row.channel_names) != channel_names:
        raise ValueError("All RuntimeRow objects must share the same channel_names order for export")
    if row.values.shape[0] != len(channel_names):
        raise ValueError("RuntimeRow values must have one row per channel")


def _write_signal_frame(path: Path, row, sampling_period_seconds: float) -> None:
    time_axis = [minute * sampling_period_seconds for minute in range(row.values.shape[1])]
    payload: dict[str, list[float]] = {"time_s": time_axis}
    for idx, channel_name in enumerate(row.channel_names):
        payload[channel_name] = row.values[idx].astype(float).tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(payload), path)
