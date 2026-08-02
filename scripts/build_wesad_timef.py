#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Build the stored WESAD ``timef`` dataset from raw subject folders.

``WESADStateQADataset`` loads its input with ``load_from_disk``, so the eval
needs a saved Arrow dataset — unlike the explorer path, it has no fallback to
raw subject folders. This writes that artifact once, so the ~minute-per-subject
preprocessing isn't repeated on every eval run.

    python scripts/build_wesad_timef.py --raw-dir data/WESAD --out data/wesad_timef

Expects the layout of the official WESAD release::

    data/WESAD/S2/S2.pkl
    data/WESAD/S2/S2_E4_Data.zip

Reading the raw data means unpickling it, which executes arbitrary code — the
row builder refuses to without an explicit trust flag. Point this only at an
archive you obtained yourself from the Siegen release.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset  # noqa: E402

from wesad.transformer import build_wesad_timef_row  # noqa: E402


def _subject_dirs(raw_dir: Path, only: set[str] | None) -> list[Path]:
    dirs = sorted(
        path for path in raw_dir.iterdir()
        if path.is_dir() and (path / f"{path.name}.pkl").exists()
    )
    if only is not None:
        dirs = [path for path in dirs if path.name in only]
        missing = only - {path.name for path in dirs}
        if missing:
            raise SystemExit(f"No subject folder for: {', '.join(sorted(missing))}")
    return dirs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw WESAD subject folders into a stored timef dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", default="data/WESAD", help="Raw WESAD subject folders.")
    parser.add_argument("--out", default="data/wesad_timef", help="Where to save the dataset.")
    parser.add_argument(
        "--subjects", default=None,
        help="Comma-separated subject ids (e.g. S2,S3). Defaults to all found.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace --out if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)

    if not raw_dir.is_dir():
        raise SystemExit(f"Raw WESAD directory not found: {raw_dir}")
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already exists — pass --overwrite to replace it.")

    only = set(args.subjects.split(",")) if args.subjects else None
    dirs = _subject_dirs(raw_dir, only)
    if not dirs:
        raise SystemExit(f"No WESAD subject folders (S*/S*.pkl) under {raw_dir}")

    print(f"Subjects : {len(dirs)} ({', '.join(path.name for path in dirs)})")
    print(f"Output   : {out_dir}")
    print()

    def rows():
        # A generator, not a list: one row carries the full 700 Hz label track
        # (~4.5M values) as Python lists, so materialising every subject at once
        # costs gigabytes. from_generator spills each row to Arrow as it goes.
        for i, subject_dir in enumerate(dirs, start=1):
            print(f"[{i}/{len(dirs)}] {subject_dir.name} …", flush=True)
            row = build_wesad_timef_row(subject_dir, trust=True)
            # "data" duplicates "values" verbatim; dropping it halves the largest
            # array in the row. Readers already prefer "values" and fall back.
            row.pop("data", None)
            yield row

    dataset = Dataset.from_generator(rows)
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    dataset.save_to_disk(str(out_dir))

    print()
    print(f"Wrote {len(dataset)} rows to {out_dir}")
    print(f"Channels: {len(dataset[0]['channel_names'])}")


if __name__ == "__main__":
    main()
