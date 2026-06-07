"""Quick schema check for the daily MHC dataset on Sherlock."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from datasets import load_from_disk

path = os.environ.get("MHC_DATASET_DIR", "")
if not path:
    print("Set MHC_DATASET_DIR"); sys.exit(1)

ds = load_from_disk(path)
row0 = ds[0]
print("columns:", list(row0.keys()))
print("total rows:", len(ds))

vals = np.asarray(row0["values"]) if "values" in row0 else None
if vals is not None:
    print("values shape:", vals.shape)

print("\nSearching for first row with Watch HR (channel 5)...")
for i, row in enumerate(ds):
    v = np.asarray(row.get("values", []))
    if v.ndim == 2 and np.any(v[5] > 0):
        print(f"  row {i}, user: {row['user_id']}")
        print(f"  HR (bpm): {v[5][v[5] > 0][:5] * 60}")
        print(f"  steps (ch3): {v[3][v[3] > 0][:5]}")
        break
    if i >= 2000:
        print("  no HR in first 2000 rows — check channel index or data")
        break
