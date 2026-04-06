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
    from extractors.statistical import StatisticalExtractor
    from extractors.structural import StructuralExtractor
    from models.local import LocalConfig, LocalModel
    from extractors.semantic import SemanticExtractor
    from visualizer import plot_row
    import numpy as np

    dataset = MHCDataset(min_wear_pct=0.0)
    annotator = Annotator([
        StatisticalExtractor(MHC_CHANNEL_CONFIG),
        StructuralExtractor(MHC_CHANNEL_CONFIG),
        SemanticExtractor(MHC_CHANNEL_CONFIG),
    ])
    model = LocalModel(
        LocalConfig(model="google/gemma-3-4b-it"),
        MHC_CHANNEL_CONFIG,
    )
    reviewer = Reviewer(model)
    captionizer = Captionizer(dataset, MHCTransformer(), annotator, reviewer=reviewer)
    print(f"Dataset size: {len(dataset)}")

    result, evaluation = captionizer.run(max_rows=5)
    print(f"Rows: {len(result.rows)}")
    print(f"Annotations: {sum(len(row.annotations) for row in result.rows)}")
    if evaluation:
        print(f"Evaluation: {len(evaluation.scores)} scores, mean={evaluation.mean_score}")

    shown = 0
    for row in result.iter_rows():
        if row.active_channel_count() >= 9:
            plot_row(row)
            shown += 1
            if shown >= 4:
                break
