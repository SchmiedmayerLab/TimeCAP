#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Demo: run clinical synthesizers on a few recordings and show grouped captions.

Usage (Sherlock):
    python scripts/demo_caption_groups.py \
        --max_users 5 \
        --split_file /path/to/weekly_labels_lookup_stride7.parquet \
        --labels cardiovascular_disease "Heart Failure or CHF" sleep_diagnosis1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical.caption_grouper import SIGNAL_LABEL_RELEVANCE, format_grouped_captions, group_captions
from clinical.data import DAILY_CLINICAL_CONFIG, iter_daily_recordings
from clinical.cadence import CadenceSynthesizer
from clinical.circadian_dipping import CircadianDippingSynthesizer
from clinical.gait_speed import GaitSpeedSynthesizer
from clinical.hr_inversion import HRInversionSynthesizer
from clinical.hr_recovery import HRRecoverySynthesizer
from clinical.hr_spread import HRSpreadSynthesizer
from clinical.nocturnal_activity import NocturnalActivitySynthesizer
from clinical.nocturnal_rhr import NocturnalRHRSynthesizer
from clinical.peak_hr_proxy import PeakHRProxySynthesizer
from clinical.resting_hr import RestingHRSynthesizer
from clinical.sedentary_burden import SedentaryBurdenSynthesizer
from clinical.sustained_bout import SustainedBoutSynthesizer

SYNTHESIZERS = [
    ("signal_3_resting_hr",          RestingHRSynthesizer()),
    ("signal_1_hr_recovery",         HRRecoverySynthesizer()),
    ("signal_17_sustained_bout",     SustainedBoutSynthesizer()),
    ("signal_16_hr_spread",          HRSpreadSynthesizer()),
    ("signal_21_circadian_dipping",  CircadianDippingSynthesizer()),
    ("signal_14_nocturnal_activity", NocturnalActivitySynthesizer()),
    ("signal_20_nocturnal_rhr",      NocturnalRHRSynthesizer()),
    ("signal_22_gait_speed",         GaitSpeedSynthesizer()),
    ("signal_18_cadence",            CadenceSynthesizer()),
    ("signal_24_sedentary_burden",   SedentaryBurdenSynthesizer()),
    ("signal_7_hr_inversion",        HRInversionSynthesizer()),
]

# Weekly synthesizers use hourly data; skip in this daily demo.
# signal_11_peak_hr_proxy and signal_6_exercise_volume are weekly-only.


def _load_split_users(split_file: str) -> set[str]:
    path = Path(split_file)
    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return set(df["user_id"].dropna().unique().tolist())
    import json
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return set(data)
    return set(data.get("train", [])) | set(data.get("validation", [])) | set(data.get("test", []))


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo grouped captions on MHC daily recordings.")
    parser.add_argument("--max_users", type=int, default=5,
                        help="Number of unique users to sample.")
    parser.add_argument("--max_recordings", type=int, default=3,
                        help="Max recordings to show per user.")
    parser.add_argument("--split_file", default=None,
                        help="Filter to sharable users (parquet with user_id or JSON split).")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Labels to show. Default: all labels in relevance map.")
    args = parser.parse_args()

    split_users = _load_split_users(args.split_file) if args.split_file else None
    target_labels = args.labels  # None → all labels

    seen_users: dict[str, int] = {}  # user_id → recordings shown

    for rec in iter_daily_recordings(split_users=split_users, max_users=args.max_users):
        uid = rec.user_id
        if seen_users.get(uid, 0) >= args.max_recordings:
            continue
        seen_users[uid] = seen_users.get(uid, 0) + 1

        # Run all synthesizers
        signal_results = {
            name: synth.synthesize(rec, DAILY_CLINICAL_CONFIG)
            for name, synth in SYNTHESIZERS
        }

        n_fired = sum(1 for anns in signal_results.values() if anns)
        print(f"\n{'='*70}")
        print(f"User: {uid}  |  Date: {rec.date}  |  Signals fired: {n_fired}/{len(SYNTHESIZERS)}")
        print(f"{'='*70}")

        grouped = group_captions(signal_results, labels=target_labels)
        output = format_grouped_captions(grouped)

        if output:
            print(output)
        else:
            print("(no captions for the selected labels)")


if __name__ == "__main__":
    main()
