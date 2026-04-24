#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Quick smoke test for label lookup + metabolic QA dataset.

Usage:
    python scripts/verify_label_lookup.py \
        --export-dir exports \
        --labels-parquet path/to/weekly_labels_lookup.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time_series_datasets.mhc_label_lookup import LabelLookup, LABEL_TYPES
from time_series_datasets.mhc_metabolic_qa_dataset import MHCMetabolicQADataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--labels-parquet", required=True)
    parser.add_argument("--target-label", default="Diabetes")
    args = parser.parse_args()

    print(f"Loading label lookup from {args.labels_parquet}")
    lookup = LabelLookup(args.labels_parquet)
    print(f"  date column: {lookup.date_column}")
    print(f"  samples: {lookup.num_samples}")
    print(f"  labels: {len(lookup.label_names)}")

    print(f"\nLoading MHCMetabolicQADataset (target_label={args.target_label})")
    train = MHCMetabolicQADataset(
        "train", "",
        target_label=args.target_label,
        export_dir=args.export_dir,
        label_lookup_path=args.labels_parquet,
    )
    val = MHCMetabolicQADataset(
        "validation", "",
        target_label=args.target_label,
        export_dir=args.export_dir,
        label_lookup_path=args.labels_parquet,
    )
    test = MHCMetabolicQADataset(
        "test", "",
        target_label=args.target_label,
        export_dir=args.export_dir,
        label_lookup_path=args.labels_parquet,
    )
    print(f"  train: {len(train)}  val: {len(val)}  test: {len(test)}")

    if len(train) > 0:
        sample = train[0]
        print(f"\nFirst train sample:")
        print(f"  answer: {sample['answer']}")
        print(f"  post_prompt: {sample['post_prompt']}")
        print(f"  num time series: {len(sample['time_series'])}")

    print("\nOK")


if __name__ == "__main__":
    main()
