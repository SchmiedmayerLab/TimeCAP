#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Run the Composer on one local MHC sample with a local Hugging Face model.

Usage:
    conda run -n AML python scripts/test_composer_local.py --index 0
    conda run -n AML python scripts/test_composer_local.py --variant daily --index 12
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from annotator import Annotator
from composer import Composer
from extractors.semantic import SemanticExtractor
from extractors.structural import StructuralExtractor


def _build_daily(min_wear_pct: float):
    from mhc.constants import MHC_CHANNEL_CONFIG
    from mhc.dataset import MHCDataset
    from mhc.transformer import MHCTransformer

    return MHCDataset(min_wear_pct=min_wear_pct), MHCTransformer(), MHC_CHANNEL_CONFIG


def _build_weekly(min_valid_hours: int):
    from mhc_weekly.constants import WEEKLY_CHANNEL_CONFIG
    from mhc_weekly.dataset import WeeklyMHCDataset
    from mhc_weekly.transformer import WeeklyMHCTransformer

    return (
        WeeklyMHCDataset(min_valid_hours=min_valid_hours),
        WeeklyMHCTransformer(),
        WEEKLY_CHANNEL_CONFIG,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Composer on one local sample.")
    parser.add_argument("--variant", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--index", type=int, default=None,
                        help="Dataset row index to inspect. If omitted, choose one at random.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed used when --index is omitted.")
    parser.add_argument("--min_wear_pct", type=float, default=0.0,
                        help="Daily dataset filter.")
    parser.add_argument("--min_valid_hours", type=int, default=0,
                        help="Weekly dataset filter.")
    parser.add_argument("--model", default="google/gemma-3-4b-it",
                        help="Local Hugging Face image-text-to-text model.")
    parser.add_argument("--device", default="mps",
                        help="Torch device. Use mps for Apple Silicon, or cpu as fallback.")
    parser.add_argument("--torch_dtype", default="bfloat16",
                        help="Torch dtype name, e.g. bfloat16 or float32.")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum generated tokens for the composed annotation.")
    parser.add_argument("--replace-source", action="store_true",
                        help="Print only the composed annotation after generation.")
    args = parser.parse_args()

    if args.variant == "daily":
        dataset, transformer, channel_config = _build_daily(args.min_wear_pct)
    else:
        dataset, transformer, channel_config = _build_weekly(args.min_valid_hours)

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty with the selected filters.")

    index = args.index
    if index is None:
        rng = random.Random(args.seed)
        index = rng.randrange(len(dataset))
    if index < 0 or index >= len(dataset):
        raise IndexError(f"--index must be in [0, {len(dataset) - 1}], got {index}.")

    annotator = Annotator([
        StructuralExtractor(channel_config),
        SemanticExtractor(channel_config),
    ])

    from models.local import LocalConfig, LocalModel

    model = LocalModel(
        LocalConfig(
            model=args.model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            max_tokens=args.max_tokens,
        ),
        channel_config,
    )
    composer = Composer(model)

    row = transformer.transform_row(dataset[index])
    row.annotations.extend(annotator.annotate(row))
    refined_annotations = composer.distill_recording(row)
    if refined_annotations:
        if args.replace_source:
            row.annotations = refined_annotations
        else:
            row.annotations.extend(refined_annotations)

    print(f"Dataset size: {len(dataset)}")
    print(f"Selected index: {index}")
    print(f"Row id: {row.row_id}")
    print(f"Date: {row.date}")
    print(f"Annotations: {len(row.annotations)}")

    for annotation in row.annotations:
        channels = ", ".join(row.display_names[idx] for idx in annotation.channel_idxs)
        channel_text = f" ({channels})" if channels else ""
        print(f"\n[{annotation.caption_type}]{channel_text}")
        print(annotation.text)


if __name__ == "__main__":
    main()
