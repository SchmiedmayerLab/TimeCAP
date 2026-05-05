#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Render publication-style caption visualizations for MHC rows.

Examples:
    python scripts/render_caption_visualization.py --row-index 2697 --example walking --out figures/walking.png
    python scripts/render_caption_visualization.py --row-index 330 --example sleep --out figures/sleep.pdf
    python scripts/render_caption_visualization.py --find --require-types statistical structural semantic cross_channel
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root or scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from annotator import Annotator
from extractors.cross_channel import CrossChannelExtractor
from extractors.semantic import SemanticExtractor
from extractors.statistical import StatisticalExtractor
from extractors.structural import StructuralExtractor
from mhc.constants import MHC_CHANNEL_CONFIG
from mhc.dataset import MHCDataset
from mhc.transformer import MHCTransformer
from synthesizers.cardio import CardioSynthesizer, EnduranceActivitySynthesizer
from synthesizers.locomotion import WalkingSynthesizer
from synthesizers.mind_body import MindBodyActivitySynthesizer
from synthesizers.other_activity import OtherActivitySynthesizer
from synthesizers.sleep import SleepSynthesizer
from synthesizers.stationary_activity import StationaryActivitySynthesizer
from visualizer import DEFAULT_PIPELINE_CAPTION_TYPES, plot_caption_pipeline_row


ALL_CAPTION_TYPES = ("statistical", "structural", "semantic", "cross_channel")


def build_mhc_annotator() -> Annotator:
    return Annotator([
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


def annotation_type_counts(row) -> dict[str, int]:
    return {caption_type: sum(1 for annotation in row.annotations if annotation.caption_type == caption_type)
            for caption_type in ALL_CAPTION_TYPES}


def find_matching_rows(
    dataset: MHCDataset,
    transformer: MHCTransformer,
    annotator: Annotator,
    *,
    require_types: tuple[str, ...],
    start: int,
    end: int,
    limit: int,
) -> list[tuple[int, object, dict[str, int]]]:
    matches = []
    for row_index in range(start, min(end, len(dataset))):
        row = transformer.transform_row(dataset[row_index])
        row.annotations.extend(annotator.annotate(row))
        counts = annotation_type_counts(row)
        if all(counts.get(caption_type, 0) > 0 for caption_type in require_types):
            matches.append((row_index, row, counts))
            if len(matches) >= limit:
                break
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a publication-style MHC caption visualization.")
    parser.add_argument("--row-index", type=int, help="MHC dataset row index to render.")
    parser.add_argument("--out", help="Output figure path, e.g. figures/example.png or .pdf.")
    parser.add_argument("--example", choices=("walking", "sleep", "auto"), default="auto",
                        help="Channel-selection preset.")
    parser.add_argument("--caption-types", nargs="+", default=list(DEFAULT_PIPELINE_CAPTION_TYPES),
                        choices=("statistical", "structural", "semantic", "cross_channel"),
                        help="Annotation types to show as caption cards.")
    parser.add_argument("--max-cards", type=int, default=6, help="Maximum caption cards to render.")
    parser.add_argument("--include-statistical", action="store_true",
                        help="Include statistical caption cards in addition to selected caption types.")
    parser.add_argument("--channels", nargs="+",
                        help="Optional channel names or visualizer aliases to render, e.g. WHR WSteps WDist WEnergy HIIT.")
    parser.add_argument("--min-wear-pct", type=float, default=0.0,
                        help="Optional MHC wear-percentage filter before indexing.")
    parser.add_argument("--find", action="store_true",
                        help="Search for rows matching required caption types instead of rendering one row.")
    parser.add_argument("--require-types", nargs="+", default=list(ALL_CAPTION_TYPES),
                        choices=ALL_CAPTION_TYPES,
                        help="Caption types required when using --find.")
    parser.add_argument("--start", type=int, default=0, help="Start row index for --find.")
    parser.add_argument("--end", type=int, default=None, help="End row index for --find, exclusive.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum rows to print for --find.")
    args = parser.parse_args()

    dataset = MHCDataset(min_wear_pct=args.min_wear_pct)
    transformer = MHCTransformer()
    annotator = build_mhc_annotator()

    if args.find:
        end = args.end if args.end is not None else len(dataset)
        matches = find_matching_rows(
            dataset,
            transformer,
            annotator,
            require_types=tuple(args.require_types),
            start=max(args.start, 0),
            end=end,
            limit=max(args.limit, 1),
        )
        print(f"Found {len(matches)} rows requiring {', '.join(args.require_types)}")
        for row_index, row, counts in matches:
            counts_text = ", ".join(f"{caption_type}={counts[caption_type]}" for caption_type in ALL_CAPTION_TYPES)
            labels = sorted({str(annotation.label) for annotation in row.annotations if annotation.label is not None})
            labels_text = ", ".join(labels[:8])
            if len(labels) > 8:
                labels_text += ", ..."
            print(f"row={row_index} date={row.date} user={row.user_id} {counts_text} labels=[{labels_text}]")
        return

    if args.row_index is None or args.out is None:
        parser.error("--row-index and --out are required unless --find is set")

    if args.row_index < 0 or args.row_index >= len(dataset):
        raise IndexError(f"row-index {args.row_index} outside dataset range [0, {len(dataset)})")

    row = transformer.transform_row(dataset[args.row_index])
    row.annotations.extend(annotator.annotate(row))

    plot_caption_pipeline_row(
        row,
        save_path=args.out,
        example=args.example,
        channel_names=args.channels,
        caption_types=tuple(args.caption_types),
        max_caption_cards=args.max_cards,
        include_statistical=args.include_statistical,
        title=f"MHC captioning pipeline | row {args.row_index} | {row.date}",
    )
    print(f"Saved caption visualization to {args.out}")


if __name__ == "__main__":
    main()
