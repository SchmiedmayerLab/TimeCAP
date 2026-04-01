#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from pathlib import Path

import pandas as pd

from timef.io import TimeFWriter
from timef.result import (
    Annotation,
    CaptionResult,
    RuntimeManifest,
    Sample,
    SampleRef,
    SampleSignalRef,
    Signal,
)
from timef.runtime import CaptionData, SignalData
from timef.schema import (
    Annotation as StoredAnnotation,
    AnnotationSampleRef,
    DatasetManifest,
    Sample as StoredSample,
    SampleSignalRef as StoredSampleSignalRef,
    Signal as StoredSignal,
)
from timef.validate import mark_validated, validate_dataset
from transformer import Transformer
from annotator import Annotator


class Captionizer:
    def __init__(self, dataset, transformer: Transformer, annotator: Annotator) -> None:
        self.dataset = dataset
        self.transformer = transformer
        self.annotator = annotator

    def run(self, max_rows: int | None = None) -> CaptionResult:
        manifest = RuntimeManifest(
            signal_specs=self.transformer.get_signal_specs(),
            annotation_specs=self.annotator.get_annotation_specs(),
        )

        n = min(len(self.dataset), max_rows) if max_rows is not None else len(self.dataset)

        # Accumulate lightweight runtime types — no schema objects during processing
        all_signals: dict[str, SignalData] = {}
        all_captions: list[CaptionData] = []

        for i in range(n):
            row = self.dataset[i]
            sigs = self.transformer.transform_row(row)
            for sig in sigs:
                all_signals[sig.id] = sig
            all_captions.extend(self.annotator.annotate(sigs))

        # Convert lightweight runtime records into the in-memory result used by plotting/debugging.
        result = CaptionResult(manifest=manifest)

        for sig in all_signals.values():
            result.signals[sig.id] = Signal(
                id=sig.id, spec_id=sig.spec_id, name=sig.name,
                data=sig.data, metadata=sig.metadata,
            )

        for cap in all_captions:
            result.samples.append(Sample(
                id=cap.sample_id,
                windows=[cap.window] if cap.window is not None else None,
                signals=[SampleSignalRef(signal_id=cap.signal_id)],
            ))
            result.annotations.append(Annotation(
                id=cap.sample_id,
                spec_id=cap.spec_id,
                samples=[SampleRef(sample_id=cap.sample_id)],
                answer=cap.answer,
            ))

        return result

    def write_timef(
        self,
        result: CaptionResult,
        root: str | Path,
        dataset_id: str | None = None,
        version: str = "0.1.0",
        source: str = "SensorTSLM",
        description: str = "",
        domains: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> DatasetManifest:
        root = Path(root)
        writer = TimeFWriter(root)

        dataset_id = dataset_id or self.dataset.__class__.__name__.removesuffix("Dataset").lower()
        domains = list(domains) if domains is not None else ["health"]
        tags = list(tags) if tags is not None else [dataset_id]

        runtime_signal_ids = list(result.signals.keys())
        signal_id_map = {signal_id: idx for idx, signal_id in enumerate(runtime_signal_ids)}

        for stored_id, runtime_signal_id in enumerate(runtime_signal_ids):
            runtime_signal = result.signals[runtime_signal_id]
            signal_file = f"signal-{stored_id:05d}.parquet"
            frame = pd.DataFrame({
                "time_s": pd.Series(range(len(runtime_signal.data)), dtype="float64") * 60.0,
                "value": runtime_signal.data,
            })
            writer.write_signal_frame(signal_file, frame)

        stored_samples: list[StoredSample] = []
        stored_annotations: list[StoredAnnotation] = []
        for sample_idx, (runtime_sample, runtime_annotation) in enumerate(zip(result.samples, result.annotations)):
            stored_samples.append(StoredSample(
                id=sample_idx,
                dataset_id=dataset_id,
                unit_timestamp="s",
                windows=(
                    [[float(start) * 60.0, float(end) * 60.0] for start, end in runtime_sample.windows]
                    if runtime_sample.windows is not None else None
                ),
                signals=[
                    StoredSampleSignalRef(
                        signal_id=signal_id_map[sig_ref.signal_id],
                        sampling_rate=sig_ref.sampling_rate or (1 / 60),
                        channels=["value"] if sig_ref.channels is None else sig_ref.channels,
                    )
                    for sig_ref in runtime_sample.signals
                ],
            ))
            stored_annotations.append(StoredAnnotation(
                id=sample_idx,
                dataset_id=dataset_id,
                spec_id=runtime_annotation.spec_id,
                samples=[
                    AnnotationSampleRef(
                        sample=sample_idx,
                        reference=sample_ref.reference,
                        forecast=sample_ref.forecast,
                    )
                    for sample_ref in runtime_annotation.samples
                ],
                question=runtime_annotation.question,
                answer=runtime_annotation.answer,
                label=runtime_annotation.label,
            ))

        stored_signals: list[StoredSignal] = []
        for runtime_signal_id, stored_id in signal_id_map.items():
            runtime_signal = result.signals[runtime_signal_id]
            signal_file = f"signal-{stored_id:05d}.parquet"
            stored_signals.append(StoredSignal(
                id=stored_id,
                spec_id=runtime_signal.spec_id,
                name=runtime_signal.name,
                sampling_rate=float(runtime_signal.metadata.get("sampling_rate_hz", 1 / 60)),
                total_duration=float(len(runtime_signal.data) * 60),
                shard_file=signal_file,
                row_group_id=0,
                row_start=0,
                row_count=len(runtime_signal.data),
            ))

        manifest = DatasetManifest(
            dataset_id=dataset_id,
            version=version,
            source=source,
            description=description,
            domains=domains,
            tags=tags,
            signals=stored_signals,
            signal_spec=result.manifest.signal_specs,
            sensor_spec=[],
            annotation_spec=result.manifest.annotation_specs,
            sample_count=len(stored_samples),
            annotation_count=len(stored_annotations),
            validated=False,
        )

        writer.write_manifest(manifest)
        writer.write_samples(stored_samples)
        writer.write_annotations(stored_annotations, manifest)
        validate_dataset(root)
        mark_validated(root, validated=True)
        return manifest.model_copy(update={"validated": True})


if __name__ == "__main__":
    from mhc.dataset import MHCDataset
    from mhc.transformer import MHCTransformer
    from mhc.constants import MHC_CHANNEL_CONFIG
    from extractors.statistical import StatisticalExtractor
    from extractors.structural import StructuralExtractor
    from visualizer import plot_row

    dataset = MHCDataset(min_wear_pct=0.0)
    annotator = Annotator([
        StatisticalExtractor(MHC_CHANNEL_CONFIG),
        StructuralExtractor(MHC_CHANNEL_CONFIG),
    ])
    captionizer = Captionizer(dataset, MHCTransformer(), annotator)
    print(f"Dataset size: {len(dataset)}")

    result = captionizer.run(max_rows=100)
    print(f"Signals: {len(result.signals)}")
    print(f"Samples: {len(result.samples)}")
    print(f"Annotations: {len(result.annotations)}")

    export_dir = Path("artifacts/timef")
    manifest = captionizer.write_timef(result, export_dir, dataset_id="mhc")
    print(f"Exported TimeF bundle to {export_dir} (validated={manifest.validated})")

    # Plot first 4 rows with at least 5 active channels
    shown = 0
    for signals, samples, annotations in result.iter_rows():
        active = sum(
            s.metadata.get("has_any_data", True) for s in signals
        )
        if active >= 5:
            plot_row(signals, samples, annotations)
            shown += 1
            if shown >= 4:
                break
