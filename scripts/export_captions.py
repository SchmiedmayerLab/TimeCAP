#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Run the caption pipeline over MHCDataset and write lean Arrow shards.

Usage:
    python scripts/export_captions.py --out exports/lean_full
    python scripts/export_captions.py --out exports/lean_test --max_rows 50
    python scripts/export_captions.py --variant daily --clinical --min_clinical_signals 5 --out exports/clinical_daily
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
from extractors.semantic import SemanticExtractor
from extractors.structural import StructuralExtractor
from mhc.cross_channel import default_extractor
from timef.schema import CaptionResult


def _build_daily(min_wear_pct: float, split_file: str | None):
    from mhc.constants import MHC_CHANNEL_CONFIG
    from mhc.dataset import MHCDataset
    from mhc.transformer import MHCTransformer
    return (
        MHCDataset(min_wear_pct=min_wear_pct, split_file=split_file),
        MHCTransformer(),
        MHC_CHANNEL_CONFIG,
    )


def _build_weekly(min_valid_hours: int, split_file: str | None):
    from mhc_weekly.constants import WEEKLY_CHANNEL_CONFIG
    from mhc_weekly.dataset import WeeklyMHCDataset
    from mhc_weekly.transformer import WeeklyMHCTransformer
    return (
        WeeklyMHCDataset(min_valid_hours=min_valid_hours, split_file=split_file),
        WeeklyMHCTransformer(),
        WEEKLY_CHANNEL_CONFIG,
    )


def main():
    parser = argparse.ArgumentParser(description="Export caption shards from MHC.")
    parser.add_argument("--variant", choices=["daily", "weekly"], default="weekly",
                        help="Which MHC dataset to caption.")
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
    parser.add_argument("--split_file", default=None,
                        help="MHC-benchmark canonical split JSON. When set, "
                             "non-sharable users are dropped at load time.")
    parser.add_argument("--clinical", action="store_true",
                        help="(daily only) Also run clinical synthesizers and append their annotations.")
    parser.add_argument("--min_clinical_signals", type=int, default=0,
                        help="(daily + --clinical) Drop rows where fewer than N clinical signals fired.")
    args = parser.parse_args()

    if args.clinical and args.variant != "daily":
        parser.error("--clinical is only supported with --variant daily")

    # MHCDataset only supports JSON split files; load parquet user IDs ourselves
    # and filter in the processing loop instead.
    split_users: set[str] | None = None
    dataset_split_file = args.split_file
    if args.split_file and Path(args.split_file).suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(args.split_file)
        split_users = set(df["user_id"].dropna().unique().tolist())
        dataset_split_file = None
        print(f"Loaded {len(split_users)} users from {args.split_file}")

    if args.variant == "daily":
        dataset, transformer, channel_config = _build_daily(
            args.min_wear_pct, dataset_split_file
        )
    else:
        dataset, transformer, channel_config = _build_weekly(
            args.min_valid_hours, dataset_split_file
        )

    annotator = Annotator([
        StructuralExtractor(channel_config),
        SemanticExtractor(channel_config),
        default_extractor(channel_config),
    ])

    clinical_synthesizers = []
    if args.clinical:
        from clinical.cadence import CadenceSynthesizer
        from clinical.circadian_dipping import CircadianDippingSynthesizer
        from clinical.data import DAILY_CLINICAL_CONFIG
        from clinical.gait_speed import GaitSpeedSynthesizer
        from clinical.hr_inversion import HRInversionSynthesizer
        from clinical.hr_recovery import HRRecoverySynthesizer
        from clinical.hr_spread import HRSpreadSynthesizer
        from clinical.nocturnal_activity import NocturnalActivitySynthesizer
        from clinical.nocturnal_rhr import NocturnalRHRSynthesizer
        from clinical.resting_hr import RestingHRSynthesizer
        from clinical.sedentary_burden import SedentaryBurdenSynthesizer
        from clinical.sustained_bout import SustainedBoutSynthesizer
        clinical_synthesizers = [
            RestingHRSynthesizer(),
            HRRecoverySynthesizer(),
            SustainedBoutSynthesizer(),
            HRSpreadSynthesizer(),
            CircadianDippingSynthesizer(),
            NocturnalActivitySynthesizer(),
            NocturnalRHRSynthesizer(),
            GaitSpeedSynthesizer(),
            CadenceSynthesizer(),
            SedentaryBurdenSynthesizer(),
            HRInversionSynthesizer(),
        ]

    n_total = len(dataset)
    start = args.start
    end = min(args.end if args.end is not None else n_total, n_total)
    if args.max_rows is not None:
        end = min(end, start + args.max_rows)

    print(f"Dataset size: {n_total}; processing rows [{start}, {end})")

    result = CaptionResult()
    failed: list[int] = []
    skipped_sparse = 0

    pbar = tqdm(range(start, end), desc="Captioning", dynamic_ncols=True)
    for i in pbar:
        try:
            if split_users is not None and dataset[i].get("user_id") not in split_users:
                continue
            row = transformer.transform_row(dataset[i])
            row.annotations.extend(annotator.annotate(row))

            if clinical_synthesizers:
                synth_results = [synth.synthesize(row, DAILY_CLINICAL_CONFIG) for synth in clinical_synthesizers]
                n_fired = sum(1 for anns in synth_results if anns)
                if args.min_clinical_signals > 0 and n_fired < args.min_clinical_signals:
                    skipped_sparse += 1
                    pbar.set_postfix(skipped=skipped_sparse)
                    continue
                for anns in synth_results:
                    row.annotations.extend(anns)

            result.rows.append(row)
        except Exception as exc:
            failed.append(i)
            pbar.set_postfix(failed=len(failed), last_err=type(exc).__name__)

    print(f"Captioned {len(result.rows)} rows, "
          f"{sum(len(r.annotations) for r in result.rows)} annotations, "
          f"{len(failed)} failed"
          + (f", {skipped_sparse} skipped (< {args.min_clinical_signals} clinical signals)" if skipped_sparse else ""))

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
