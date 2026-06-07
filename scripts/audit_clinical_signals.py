#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Audit clinician-derived signal prevalence over MHC dataset.

Runs each clinical signal detector as a CrossChannelSynthesizer and reports
how often each fires (window-level and participant-level).

Usage:
    python scripts/audit_clinical_signals.py --variant daily --max_rows 50
    python scripts/audit_clinical_signals.py --variant both --out audit_out/results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical.auditor import ClinicalSignalAuditor
from clinical.bradycardia import NocturnalBradycardiaSynthesizer
from clinical.cadence import CadenceSynthesizer
from clinical.circadian_dipping import CircadianDippingSynthesizer
from clinical.exercise_volume import ExerciseVolumeSynthesizer
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


DAILY_SYNTHESIZERS = [
    ("signal_3_resting_hr",          RestingHRSynthesizer()),
    ("signal_20_nocturnal_rhr",      NocturnalRHRSynthesizer()),
    ("signal_12_bradycardia",        NocturnalBradycardiaSynthesizer()),
    ("signal_14_nocturnal_activity", NocturnalActivitySynthesizer()),
    ("signal_16_hr_spread",          HRSpreadSynthesizer()),
    ("signal_17_sustained_bout",     SustainedBoutSynthesizer()),
    ("signal_18_cadence",            CadenceSynthesizer()),
    ("signal_21_circadian_dipping",  CircadianDippingSynthesizer()),
    ("signal_22_gait_speed",         GaitSpeedSynthesizer()),
    ("signal_24_sedentary_burden",   SedentaryBurdenSynthesizer()),
    ("signal_1_hr_recovery",         HRRecoverySynthesizer()),
    ("signal_7_hr_inversion",        HRInversionSynthesizer()),
]

WEEKLY_SYNTHESIZERS = [
    ("signal_6_exercise_volume",     ExerciseVolumeSynthesizer()),
    ("signal_11_peak_hr_proxy",      PeakHRProxySynthesizer()),
]

DEFERRED_SIGNALS = [
    "signal_4_sleep_regularity  — cross-day; needs multi-recording comparison",
    "signal_8_sleep_exercise_lag — cross-day lag; out of scope for per-recording synthesizers",
    "signal_23_depression        — qualitative multi-signal pattern; no binary threshold",
]


def run_daily(args, split_users: set[str] | None = None) -> tuple[ClinicalSignalAuditor, int]:
    from clinical.data import DAILY_CLINICAL_CONFIG, iter_daily_recordings

    all_user_ids: set[str] = set()

    def _gen():
        for rec in iter_daily_recordings(max_rows=args.max_rows, max_users=args.max_users, split_users=split_users):
            all_user_ids.add(rec.user_id)
            yield rec

    auditor = ClinicalSignalAuditor(DAILY_SYNTHESIZERS, DAILY_CLINICAL_CONFIG)
    auditor.run(_gen(), desc="Daily audit")
    n_participants = len(all_user_ids)

    print("\n=== DAILY (minute-level) ===")
    auditor.print_table(n_participants_total=n_participants)
    if args.examples:
        auditor.print_examples()
    return auditor, n_participants


def run_weekly(args, split_users: set[str] | None = None) -> tuple[ClinicalSignalAuditor, int]:
    from clinical.data import WEEKLY_CLINICAL_CONFIG, iter_weekly_recordings

    all_user_ids: set[str] = set()

    def _gen():
        for rec in iter_weekly_recordings(max_rows=args.max_rows, max_users=args.max_users, split_users=split_users):
            all_user_ids.add(rec.user_id)
            yield rec

    auditor = ClinicalSignalAuditor(WEEKLY_SYNTHESIZERS, WEEKLY_CLINICAL_CONFIG)
    auditor.run(_gen(), desc="Weekly audit")
    n_participants = len(all_user_ids)

    print("\n=== WEEKLY (hourly) ===")
    auditor.print_table(n_participants_total=n_participants)
    if args.examples:
        auditor.print_examples()
    return auditor, n_participants


def _load_split_users(split_file: str) -> set[str]:
    path = Path(split_file)
    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return set(df["user_id"].dropna().unique().tolist())
    else:
        import json
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return set(data)
        return set(data.get("train", [])) | set(data.get("validation", [])) | set(data.get("test", []))


def main():
    parser = argparse.ArgumentParser(description="Audit clinical signal prevalence in MHC.")
    parser.add_argument("--variant", choices=["daily", "weekly", "both"], default="both")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Cap rows per variant (for smoke tests).")
    parser.add_argument("--max_users", type=int, default=None,
                        help="Cap number of unique participants (samples across the full user range).")
    parser.add_argument("--out", default=None,
                        help="Optional CSV output path.")
    parser.add_argument("--split_file", default=None,
                        help="Filter to sharable users: JSON with train/val/test keys, "
                             "or parquet with a user_id column (e.g. weekly_labels_lookup_stride7.parquet).")
    parser.add_argument("--examples", action="store_true",
                        help="Print example captions for each signal.")
    args = parser.parse_args()

    print("\nDeferred signals (not auditable per-recording):")
    for s in DEFERRED_SIGNALS:
        print(f"  • {s}")

    split_users = _load_split_users(args.split_file) if args.split_file else None
    if split_users:
        print(f"\nFiltering to {len(split_users)} users from {args.split_file}")

    daily_auditor = weekly_auditor = None
    daily_n = weekly_n = None

    if args.variant in ("daily", "both"):
        daily_auditor, daily_n = run_daily(args, split_users=split_users)
    if args.variant in ("weekly", "both"):
        weekly_auditor, weekly_n = run_weekly(args, split_users=split_users)

    if args.out:
        out = Path(args.out)
        if daily_auditor:
            daily_out = out.with_stem(out.stem + "_daily") if args.variant == "both" else out
            daily_auditor.save_csv(daily_out, n_participants_total=daily_n)
        if weekly_auditor:
            weekly_out = out.with_stem(out.stem + "_weekly") if args.variant == "both" else out
            weekly_auditor.save_csv(weekly_out, n_participants_total=weekly_n)


if __name__ == "__main__":
    main()
