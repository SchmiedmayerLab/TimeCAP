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
from reviewer import EvaluationResult, Reviewer


class Captionizer:
    def __init__(
        self,
        dataset,
        transformer: Transformer,
        annotator: Annotator,
        reviewer: Reviewer | None = None,
    ) -> None:
        self.dataset = dataset
        self.transformer = transformer
        self.annotator = annotator
        self.reviewer = reviewer

    def run(
        self, max_rows: int | None = None,
    ) -> tuple[CaptionResult, EvaluationResult | None]:
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

        evaluation = self.reviewer.evaluate(result) if self.reviewer else None
        return result, evaluation


if __name__ == "__main__":
    import argparse

    import numpy as np
    from dotenv import load_dotenv
    load_dotenv(override=True)

    from mhc.constants import ACTIVITY_CHANNELS, SLEEP_CHANNELS
    from extractors.statistical import StatisticalExtractor
    from extractors.structural import StructuralExtractor
    from extractors.semantic import SemanticExtractor
    from models.client import ClientConfig, ClientModel
    from visualizer import plot_row

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["daily", "weekly"], default="daily")
    args = parser.parse_args()

    if args.mode == "weekly":
        from mhc_weekly.dataset import WeeklyMHCDataset
        from mhc_weekly.transformer import WeeklyMHCTransformer
        from mhc_weekly.constants import WEEKLY_CHANNEL_CONFIG

        channel_config = WEEKLY_CHANNEL_CONFIG
        dataset = WeeklyMHCDataset()
        transformer = WeeklyMHCTransformer()
        min_window = 1
    else:
        from mhc.dataset import MHCDataset
        from mhc.transformer import MHCTransformer
        from mhc.constants import MHC_CHANNEL_CONFIG

        channel_config = MHC_CHANNEL_CONFIG
        dataset = MHCDataset()
        transformer = MHCTransformer()
        min_window = 5

    annotator = Annotator([
        StatisticalExtractor(channel_config),
        StructuralExtractor(channel_config),
        SemanticExtractor(channel_config, activity_channels=ACTIVITY_CHANNELS, sleep_channels=SLEEP_CHANNELS, min_window=min_window),
    ])
    captionizer = Captionizer(dataset, transformer, annotator)
    print(f"Dataset size: {len(dataset)}")

    result, _ = captionizer.run(max_rows=1)
    print(f"Signals: {len(result.signals)}")
    print(f"Samples: {len(result.samples)}")
    print(f"Annotations: {len(result.annotations)}")

    filtered = CaptionResult(manifest=result.manifest)
    for signals, samples, annotations in result.iter_rows():
        active = sum(not np.all(np.isnan(s.data)) for s in signals)
        if active >= 1:
            for s in signals:
                filtered.signals[s.id] = s
            filtered.samples.extend(samples)
            filtered.annotations.extend(annotations)
    print(f"Filtered rows: {len(set(s.metadata.get('row_id') for s in filtered.signals.values()))}")

    model = ClientModel(
        ClientConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-5.4",
        ),
        channel_config,
    )
    reviewer = Reviewer(model)
    evaluation = reviewer.evaluate(filtered)
    print(f"Evaluation: {len(evaluation.scores)} scores, mean={evaluation.mean_score}")
    for j, s in enumerate(evaluation.scores):
        print(f"  [{j}] score={s.score}  feedback={s.feedback}")

    for i, (signals, samples, annotations) in enumerate(filtered.iter_rows()):
        plot_row(signals, samples, annotations, save_path=f"sample_plot_{i}.png")
        print(f"Saved sample_plot_{i}.png")
        if i >= 3:
            break
