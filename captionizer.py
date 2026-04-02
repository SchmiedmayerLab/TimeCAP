#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from timef.schema import DatasetManifest, CaptionResult
from transformer import Transformer
from annotator import Annotator


class Captionizer:
    def __init__(self, dataset, transformer: Transformer, annotator: Annotator) -> None:
        self.dataset = dataset
        self.transformer = transformer
        self.annotator = annotator

    def run(self, max_rows: int | None = None) -> CaptionResult:
        manifest = DatasetManifest()
        for spec in self.transformer.get_signal_specs():
            manifest.signal_specs[spec.id] = spec
        for spec in self.annotator.get_annotation_specs():
            manifest.annotation_specs[spec.id] = spec

        result = CaptionResult(manifest=manifest)
        failed_rows: list[int] = []

        for i in range(len(self.dataset)):
            if max_rows is not None and i >= max_rows:
                break

            try:
                row = self.dataset[i]
                signals = self.transformer.transform_row(row)
                samples, annotations = self.annotator.annotate(signals)

                for signal in signals:
                    result.signals[signal.id] = signal
                result.samples.extend(samples)
                result.annotations.extend(annotations)
            except Exception:
                failed_rows.append(i)

        if failed_rows:
            print(f"{len(failed_rows)} / {i + 1} rows failed during captioning: {failed_rows}")

        return result


if __name__ == "__main__":
    from mhc.dataset import MHCDataset
    from mhc.transformer import MHCTransformer
    from mhc.constants import MHC_CHANNEL_CONFIG, ACTIVITY_CHANNELS, SLEEP_CHANNELS
    from extractors.statistical import StatisticalExtractor
    from extractors.structural import StructuralExtractor
    from extractors.semantic import SemanticExtractor
    from visualizer import plot_row
    import numpy as np

    dataset = MHCDataset(min_wear_pct=0.0)
    annotator = Annotator([
        StatisticalExtractor(MHC_CHANNEL_CONFIG),
        StructuralExtractor(MHC_CHANNEL_CONFIG),
        SemanticExtractor(MHC_CHANNEL_CONFIG, activity_channels=ACTIVITY_CHANNELS, sleep_channels=SLEEP_CHANNELS),
    ])
    captionizer = Captionizer(dataset, MHCTransformer(), annotator)
    print(f"Dataset size: {len(dataset)}")

    result = captionizer.run(max_rows=1000)
    print(f"Signals: {len(result.signals)}")
    print(f"Samples: {len(result.samples)}")
    print(f"Annotations: {len(result.annotations)}")

    shown = 0
    for signals, samples, annotations in result.iter_rows():
        active = sum(
            s.metadata.get("has_any_data", True) for s in signals
        )
        if active >= 9:
            plot_row(signals, samples, annotations)
            shown += 1
            if shown >= 4:
                break
