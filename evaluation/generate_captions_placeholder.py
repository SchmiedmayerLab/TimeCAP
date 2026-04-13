#!/usr/bin/env python
"""Generate a placeholder captions.parquet for end-to-end evaluation testing.

This script creates dummy captions for all user-weeks in the weekly dev subset,
so the full caption-assisted evaluation pipeline can be tested before the
trained captionizer is available.

TODO: once the captionizer pipeline is trained (Stage 1), replace this script
with one that:
  1. Loads the trained model checkpoint.
  2. Runs each weekly recording through the captionizer.
  3. Writes the real generated captions to the same captions.parquet format.

Usage:
    python evaluation/generate_captions_placeholder.py

Output:
    data/mhc_dev_subset_captions/captions.parquet
    Columns: user_id (str), week_start (str), caption (str)

Once generated, run eval with captions:
    python evaluation/run_eval.py \\
        --dataset mhc_metabolic_biological_sex \\
        --task metabolic_biological_sex \\
        --model gpt-4o \\
        --captions-dir data/mhc_dev_subset_captions
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import datasets as hf_ds
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WEEKLY_PATH  = Path("data/mhc_dev_subset_weekly_hf")
OUTPUT_DIR   = Path("data/mhc_dev_subset_captions")
OUTPUT_FILE  = OUTPUT_DIR / "captions.parquet"

# ---------------------------------------------------------------------------
# Placeholder caption template.
# TODO: replace with real captionizer output.
# ---------------------------------------------------------------------------

def _placeholder_caption(row: dict) -> str:
    """Generate a structurally valid but content-free placeholder caption.

    The format mirrors what the trained captionizer will produce — a free-text
    summary of the week's sensor patterns.  Replace this function body with
    a call to the trained model.
    """
    n_valid = row.get("n_valid_days", "?")
    n_hours = row.get("n_valid_hours", "?")
    return (
        f"[PLACEHOLDER CAPTION — replace with trained captionizer output] "
        f"This week contains {n_valid} valid days and {n_hours} valid hours of sensor data. "
        f"Heart rate, step count, distance, and active energy patterns are visible across the week."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not WEEKLY_PATH.exists():
        logger.error(f"Weekly dataset not found: {WEEKLY_PATH}")
        logger.error("Run scripts/build_weekly_dev_subset.py first (data already built if you see data/mhc_dev_subset_weekly_hf).")
        sys.exit(1)

    logger.info(f"Loading weekly dataset from {WEEKLY_PATH} ...")
    ds = hf_ds.load_from_disk(str(WEEKLY_PATH))
    logger.info(f"  {len(ds)} rows")

    records = []
    for row in ds:
        records.append({
            "user_id":    row["user_id"],
            "week_start": row["week_start"],
            "caption":    _placeholder_caption(row),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"Saved {len(df)} captions → {OUTPUT_FILE}")
    logger.info("")
    logger.info("Run with captions:")
    logger.info(
        "  python evaluation/run_eval.py "
        "--dataset mhc_metabolic_biological_sex "
        "--task metabolic_biological_sex "
        "--model gpt-4o "
        f"--captions-dir {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
