#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import load_from_disk


def scan_distinct_users(dataset_dir: Path) -> list[str]:
    users: set[str] = set()
    shard_paths = sorted(dataset_dir.glob("data-*.arrow"))
    if not shard_paths:
        raise FileNotFoundError(f"No Arrow shards found in {dataset_dir}")

    for i, shard_path in enumerate(shard_paths, 1):
        with pa.memory_map(str(shard_path), "r") as source:
            reader = ipc.open_stream(source)
            user_idx = reader.schema.get_field_index("user_id")
            for batch in reader:
                users.update(batch.column(user_idx).to_pylist())
        if i % 50 == 0 or i == len(shard_paths):
            print(f"scanned {i}/{len(shard_paths)} shards, distinct users so far: {len(users)}")

    return sorted(users)


def select_users(all_users: list[str], subset_size: int, seed: int) -> list[str]:
    if subset_size > len(all_users):
        raise ValueError(
            f"Requested {subset_size} users, but dataset only contains {len(all_users)} distinct users."
        )
    rng = random.Random(seed)
    return sorted(rng.sample(all_users, k=subset_size))


def build_subset(dataset_dir: Path, selected_users: set[str]):
    ds = load_from_disk(str(dataset_dir))
    print(f"loaded full dataset with {len(ds)} rows")

    subset = ds.filter(
        lambda user_id: user_id in selected_users,
        input_columns=["user_id"],
    )
    print(f"filtered subset rows: {len(subset)}")
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a real-user MHC HF development subset dataset.")
    parser.add_argument(
        "--input-dir",
        default=os.environ.get("MHC_SOURCE_DATASET_DIR"),
        help="Path to the full MHC daily HF dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("MHC_DATASET_DIR", "data/mhc_dev_subset_hf"),
        help="Path where the development subset dataset should be written.",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=500,
        help="Number of distinct users to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible user sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    args = parser.parse_args()

    if not args.input_dir:
        raise ValueError(
            "Missing input dataset path. Pass --input-dir or set MHC_SOURCE_DATASET_DIR."
        )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    print(f"scanning distinct users in {input_dir}")
    all_users = scan_distinct_users(input_dir)
    print(f"found {len(all_users)} distinct users")

    selected_users = select_users(all_users, args.num_users, args.seed)
    print(f"selected {len(selected_users)} users with seed={args.seed}")

    subset = build_subset(input_dir, set(selected_users))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    subset.save_to_disk(str(output_dir))
    print(f"wrote subset dataset to {output_dir}")

    selection_path = output_dir / "selected_users.json"
    selection_path.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "num_users": args.num_users,
                "seed": args.seed,
                "selected_users": selected_users,
            },
            indent=2,
        )
    )
    print(f"wrote user selection manifest to {selection_path}")


if __name__ == "__main__":
    main()
