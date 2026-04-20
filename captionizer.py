#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from timef.schema import CaptionResult
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
        result = CaptionResult()
        failed_rows: list[int] = []

        for i in range(len(self.dataset)):
            if max_rows is not None and i >= max_rows:
                break

            try:
                row = self.transformer.transform_row(self.dataset[i])
                row.annotations.extend(self.annotator.annotate(row))
                result.rows.append(row)
            except Exception:
                failed_rows.append(i)

        if failed_rows:
            print(f"{len(failed_rows)} / {i + 1} rows failed during captioning: {failed_rows}")

        evaluation = self.reviewer.evaluate(result) if self.reviewer else None
        return result, evaluation


if __name__ == "__main__":
    from mhc.dataset import MHCDataset
    from mhc.transformer import MHCTransformer
    from mhc.constants import MHC_CHANNEL_CONFIG
    from extractors.cross_channel import CrossChannelExtractor
    from synthesizers.cardio import CardioSynthesizer, EnduranceActivitySynthesizer
    from synthesizers.locomotion import WalkingSynthesizer
    from synthesizers.mind_body import MindBodyActivitySynthesizer
    from synthesizers.other_activity import OtherActivitySynthesizer
    from synthesizers.sleep import SleepSynthesizer
    from synthesizers.stationary_activity import StationaryActivitySynthesizer
    from extractors.statistical import StatisticalExtractor
    from extractors.structural import StructuralExtractor
    from models.local import LocalConfig, LocalModel
    from models.client import ClientConfig, ClientModel
    from extractors.semantic import SemanticExtractor
    from visualizer import plot_row
    import numpy as np

    dataset = MHCDataset(min_wear_pct=0.0)
    annotator = Annotator([
        StatisticalExtractor(MHC_CHANNEL_CONFIG),
        StructuralExtractor(MHC_CHANNEL_CONFIG),
        SemanticExtractor(MHC_CHANNEL_CONFIG),
        CrossChannelExtractor(
            MHC_CHANNEL_CONFIG,
            synthesizers=[
                SleepSynthesizer(min_duration=5),
                WalkingSynthesizer(min_duration=5),
                EnduranceActivitySynthesizer(min_duration=5),
                MindBodyActivitySynthesizer(min_duration=5),
                OtherActivitySynthesizer(min_duration=5),
                StationaryActivitySynthesizer(min_duration=5),
                CardioSynthesizer(min_duration=5),
            ],
        ),
    ])

    captionizer = Captionizer(dataset, MHCTransformer(), annotator)
    print(f"Dataset size: {len(dataset)}")

    result, _ = captionizer.run(max_rows=1)
    print(f"Rows: {len(result.rows)}")
    print(f"Annotations: {sum(len(row.annotations) for row in result.rows)}")

    filtered = CaptionResult(rows=[row for row in result.iter_rows() if row.active_channel_count() >= 9])
    print(f"Filtered rows: {len(filtered.rows)}")

    model = ClientModel(
        ClientConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-5.4",
        ),
        MHC_CHANNEL_CONFIG,
    )
    reviewer = Reviewer(model)
    evaluation = reviewer.evaluate(filtered)
    print(f"Evaluation: {len(evaluation.scores)} scores, mean={evaluation.mean_score}")
    for j, s in enumerate(evaluation.scores):
        print(f"  [{j}] score={s.score}  feedback={s.feedback}")

    for i, row in enumerate(filtered.iter_rows()):
        plot_row(row, save_path=f"sample_plot_{i}.png")
        print(f"Saved sample_plot_{i}.png")
        if i >= 3:
            break
