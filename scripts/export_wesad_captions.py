#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""Run the WESAD annotation pipeline over a stored dataset and write lean Arrow shards.

Usage:
    python scripts/export_wesad_captions.py --out exports/wesad_full
    python scripts/export_wesad_captions.py --out exports/wesad_test --max_rows 50
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
from synthesizers.wesad_states import default_wesad_synthesizers
from timef.schema import CaptionResult


def _build_wesad(dataset_dir: str | None):
    from wesad.constants import WESAD_CHANNEL_CONFIG
    from wesad.dataset import WESADDataset
    from wesad.transformer import WESADTransformer

    dataset = WESADDataset(dataset_dir=dataset_dir) if dataset_dir else WESADDataset()
    return dataset, WESADTransformer(), WESAD_CHANNEL_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Export WESAD caption shards from a stored dataset.")
    parser.add_argument("--out", default="exports/wesad_full",
                        help="Output directory for Arrow shards.")
    parser.add_argument("--wesad-dataset-dir", "--wesad_dataset_dir", dest="wesad_dataset_dir", default=None,
                        help="Optional override path for the stored WESAD timef dataset.")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Cap on rows (for smoke tests).")
    parser.add_argument("--min_active_channels", type=int, default=0,
                        help="Drop rows with fewer active channels (0 = keep all).")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index (for sharded runs).")
    parser.add_argument("--end", type=int, default=None,
                        help="End index exclusive (for sharded runs).")
    args = parser.parse_args()

    dataset, transformer, channel_config = _build_wesad(args.wesad_dataset_dir)

    annotator = Annotator([
        StatisticalExtractor(channel_config),
        StructuralExtractor(channel_config),
        SemanticExtractor(channel_config),
        CrossChannelExtractor(
            channel_config,
            synthesizers=default_wesad_synthesizers(),
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

    if failed:
        preview = ", ".join(str(i) for i in failed[:10])
        raise RuntimeError(
            f"Aborting export because {len(failed)} rows failed "
            f"(first failed indices: {preview})"
        )

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
