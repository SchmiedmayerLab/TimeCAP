#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather

from timef.schema import CaptionResult, Recording


SCHEMA_VERSION = "1"


def write_caption_result(
    result: CaptionResult,
    path: Path,
    *,
    rows_per_shard: int = 4096,
    compression: str | None = "zstd",
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rows = result.rows
    if not rows:
        return
    for shard_idx, start in enumerate(range(0, len(rows), rows_per_shard)):
        shard_rows = rows[start : start + rows_per_shard]
        dicts = [_recording_to_dict(rec) for rec in shard_rows]
        table = pa.Table.from_pylist(dicts)
        table = table.replace_schema_metadata(
            {"sensortslm.schema_version": SCHEMA_VERSION}
        )
        shard_path = path / f"recordings_{shard_idx:04d}.arrow"
        feather.write_feather(table, shard_path, compression=compression)


def _recording_to_dict(rec: Recording) -> dict:
    return {
        "row_id": rec.row_id,
        "user_id": rec.user_id,
        "date": rec.date,
        "values": rec.values.tolist(),
        "channel_names": list(rec.channel_names),
        "display_names": list(rec.display_names),
        "units": list(rec.units),
        "has_any_data": rec.has_any_data.tolist(),
        "channel_variance": rec.channel_variance.tolist(),
        "total_nonwear_minutes": rec.total_nonwear_minutes,
        "wear_pct": rec.wear_pct,
        "annotations": [
            {
                "caption_type": a.caption_type,
                "text": a.text,
                "channel_idxs": list(a.channel_idxs),
                "window_start": a.window[0] if a.window else None,
                "window_end": a.window[1] if a.window else None,
                "label": None if a.label is None else str(a.label),
            }
            for a in rec.annotations
        ],
    }
