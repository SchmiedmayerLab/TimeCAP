#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Run the generic annotation pipeline over a stored dataset and write lean Arrow shards.

Usage:
    python scripts/export_captions.py --out exports/lean_full
    python scripts/export_captions.py --out exports/lean_test --max_rows 50
    python scripts/export_captions.py --variant wesad --out exports/wesad_poc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm.auto import tqdm

from annotator import Annotator
from exporters.lean import write_caption_result
from extractors.cross_channel import CrossChannelExtractor
from extractors.statistical import StatisticalExtractor
from extractors.semantic import SemanticExtractor
from extractors.structural import StructuralExtractor
from synthesizers.wesad_states import (
    AmusementActivationSynthesizer,
    RecoveryCalmSynthesizer,
    StressResponseSynthesizer,
)
from timef.schema import CaptionResult


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


def _build_wesad(dataset_dir: str | None):
    from wesad.constants import WESAD_CHANNEL_CONFIG
    from wesad.dataset import WESADDataset
    from wesad.transformer import WESADTransformer

    dataset = WESADDataset(dataset_dir=dataset_dir) if dataset_dir else WESADDataset()
    return dataset, WESADTransformer(), WESAD_CHANNEL_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Export caption shards from a stored dataset.")
    parser.add_argument("--variant", choices=["daily", "weekly", "wesad"], default="weekly",
                        help="Which dataset variant to caption.")
    parser.add_argument("--out", default="exports/lean_full",
                        help="Output directory for Arrow shards.")
    parser.add_argument("--min_wear_pct", type=float, default=0.0,
                        help="(daily) Filter by min wear percent.")
    parser.add_argument("--min_valid_hours", type=int, default=0,
                        help="(weekly) Filter by min valid hours.")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Cap on rows (for smoke tests).")
    parser.add_argument("--min_active_channels", type=int, default=0,
                        help="Drop rows with fewer active channels (0 = keep all).")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index (for sharded runs).")
    parser.add_argument("--end", type=int, default=None,
                        help="End index exclusive (for sharded runs).")
    parser.add_argument("--wesad_dataset_dir", default=None,
                        help="Optional override path for the stored WESAD timef dataset.")
    args = parser.parse_args()

    if args.variant == "daily":
        dataset, transformer, channel_config = _build_daily(args.min_wear_pct)
    elif args.variant == "wesad":
        dataset, transformer, channel_config = _build_wesad(args.wesad_dataset_dir)
    else:
        dataset, transformer, channel_config = _build_weekly(args.min_valid_hours)

    annotator = Annotator([
        StatisticalExtractor(channel_config),
        StructuralExtractor(channel_config),
        SemanticExtractor(channel_config),
        *(
            [
                CrossChannelExtractor(
                    channel_config,
                    synthesizers=[
                        StressResponseSynthesizer(min_duration=30),
                        RecoveryCalmSynthesizer(
                            label_channel="wesad:label:baseline",
                            annotation_label="calm_baseline",
                            template_key="calm_baseline",
                            min_duration=30,
                        ),
                        RecoveryCalmSynthesizer(
                            label_channel="wesad:label:meditation",
                            annotation_label="calm_meditation",
                            template_key="calm_meditation",
                            min_duration=30,
                        ),
                        AmusementActivationSynthesizer(min_duration=30),
                    ],
                )
            ]
            if args.variant == "wesad"
            else []
        ),
    ])

    n_total = len(dataset)
    start = args.start
    end = min(args.end if args.end is not None else n_total, n_total)
    if args.max_rows is not None:
        end = min(end, start + args.max_rows)

    print(f"Dataset size: {n_total}; processing rows [{start}, {end})")

    result = CaptionResult()
    failed: list[int] = []

    pbar = tqdm(range(start, end), desc="Captioning", dynamic_ncols=True)
    for i in pbar:
        try:
            row = transformer.transform_row(dataset[i])
            row.annotations.extend(annotator.annotate(row))
            result.rows.append(row)
        except Exception as exc:
            failed.append(i)
            pbar.set_postfix(failed=len(failed), last_err=type(exc).__name__)

    print(f"Captioned {len(result.rows)} rows, "
          f"{sum(len(r.annotations) for r in result.rows)} annotations, "
          f"{len(failed)} failed")

    if args.min_active_channels > 0:
        before = len(result.rows)
        result = CaptionResult(
            rows=[r for r in result.iter_rows()
                  if r.active_channel_count() >= args.min_active_channels]
        )
        print(f"Filtered (>= {args.min_active_channels} channels): "
              f"{before} -> {len(result.rows)} rows")

    out = Path(args.out)
    write_caption_result(result, out)
    print(f"Wrote shards to {out}")


if __name__ == "__main__":
    main()
