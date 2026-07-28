#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Generate LLM captions for captioned MHC weekly rows and compare to rule-based text.

Usage:
    # Preview the prompt for 3 rows (no API call):
    python scripts/export_llm_captions.py --limit 3 --dry-run

    # Generate numeric-text captions via OpenAI:
    python scripts/export_llm_captions.py --limit 5 --repr numeric --out exports/llm_numeric

    # Generate plot-based captions:
    python scripts/export_llm_captions.py --limit 5 --repr plots --out exports/llm_plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractors import ChannelConfig
from models.client import ClientConfig, ClientModel
from timef.schema import Annotation, CaptionResult, Recording

PROMPT = "Generate a detailed caption for the following time-series data:"
DEFAULT_DATA = Path("data/data-00000-of-00001.parquet")
# Only used for plot axis labels in ClientModel; no detectors/templates needed.
PLOT_CHANNEL_CONFIG = ChannelConfig(
    names=[],
    meta={},
    continuous=frozenset(),
    time_unit="hours",
)


def _decimals_for_unit(unit: str | None) -> int:
    if not unit:
        return 1
    if unit.startswith(("steps", "kcal", "count")):
        return 0
    return 1


def _serialize_channels(row: dict) -> str:
    values = np.asarray(row["values"], dtype=float)
    lines: list[str] = []
    for i, has_data in enumerate(row["has_any_data"]):
        if not has_data:
            continue
        name = row["display_names"][i]
        unit = row["units"][i]
        unit_str = f" ({unit})" if unit else ""
        decimals = _decimals_for_unit(unit)
        channel = values[i]
        formatted = ",".join(
            "NaN" if not np.isfinite(v) else f"{v:.{decimals}f}" for v in channel
        )
        lines.append(f"{name}{unit_str}, hours 0-167: {formatted}")
    return "\n".join(lines)


def _build_prompt(row: dict, repr_mode: str) -> str:
    if repr_mode == "numeric":
        payload = _serialize_channels(row)
        if payload:
            return f"{PROMPT}\n\n{payload}"
        return PROMPT
    return PROMPT


def _rule_caption(row: dict) -> str:
    return " ".join(a["text"] for a in row["annotations"] if a.get("text"))


def _row_to_recording(row: dict) -> Recording:
    values = np.asarray(row["values"], dtype=np.float32)
    has_any_data = np.asarray(row["has_any_data"], dtype=bool)
    channel_variance = np.asarray(row["channel_variance"], dtype=np.float32)
    return Recording(
        row_id=row["row_id"],
        user_id=row["user_id"],
        date=row["date"],
        values=values,
        channel_names=tuple(row["channel_names"]),
        display_names=tuple(row["display_names"]),
        units=tuple(row["units"]),
        has_any_data=has_any_data,
        channel_variance=channel_variance,
    )


def _active_signals(recording: Recording):
    return [recording.signal(i) for i, ok in enumerate(recording.has_any_data) if ok]


def _load_done_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(json.loads(line)["row_id"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM captions for MHC weekly data.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to captioned parquet (or HF dataset dir with parquet inside).",
    )
    parser.add_argument(
        "--repr",
        choices=["numeric", "plots"],
        default="numeric",
        help="numeric: channel values as text; plots: matplotlib PNGs per active channel.",
    )
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name.")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to caption.")
    parser.add_argument("--start", type=int, default=0, help="Start row index.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("exports/llm_captions"),
        help="Output directory for side_by_side.jsonl and optional arrow shards.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print assembled prompts only; do not call the API.",
    )
    parser.add_argument(
        "--write-shards",
        action="store_true",
        help="Also write LLM-caption feather shards via write_caption_result.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=10_000,
        help="ClientModel request cap (default 100 in ClientConfig is too low).",
    )
    args = parser.parse_args()

    data_path = args.data
    if data_path.is_dir():
        data_path = data_path / "data-00000-of-00001.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    table = pq.read_table(data_path)
    rows = table.slice(args.start, table.num_rows - args.start).to_pylist()
    if args.limit is not None:
        rows = rows[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out / "side_by_side.jsonl"
    done = _load_done_ids(jsonl_path)

    model = None
    if not args.dry_run:
        config = ClientConfig(
            model=args.model,
            temperature=0.0,
            max_requests=args.max_requests,
        )
        model = ClientModel(config=config, channel_config=PLOT_CHANNEL_CONFIG)

    caption_result = CaptionResult()
    n_new = 0

    for row in rows:
        row_id = row["row_id"]
        if row_id in done:
            continue

        recording = _row_to_recording(row)
        prompt = _build_prompt(row, args.repr)
        rule_text = _rule_caption(row)

        if args.dry_run:
            print("=" * 72)
            print(f"row_id: {row_id}")
            print(f"user_id: {row['user_id']}  date: {row['date']}")
            print(f"active channels: {int(recording.has_any_data.sum())} / {len(recording.has_any_data)}")
            print(f"repr: {args.repr}")
            print("-" * 72)
            if args.repr == "plots":
                print(f"[would attach {len(_active_signals(recording))} plot images, then this text:]")
            print(prompt[:4000])
            if len(prompt) > 4000:
                print(f"... ({len(prompt) - 4000} more chars)")
            print()
            continue

        assert model is not None
        if args.repr == "numeric":
            response = model.process(prompt=prompt, signals=[])
        else:
            response = model.process(
                prompt=prompt,
                signals=_active_signals(recording),
            )

        llm_text = response.text.strip()
        record = {
            "row_id": row_id,
            "user_id": row["user_id"],
            "date": row["date"],
            "repr": args.repr,
            "model": args.model,
            "n_active_channels": int(recording.has_any_data.sum()),
            "rule_caption": rule_text,
            "llm_caption": llm_text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }
        with jsonl_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        if args.write_shards:
            active_idxs = tuple(
                i for i, ok in enumerate(recording.has_any_data) if ok
            )
            out_rec = Recording(
                row_id=recording.row_id,
                user_id=recording.user_id,
                date=recording.date,
                values=recording.values,
                channel_names=recording.channel_names,
                display_names=recording.display_names,
                units=recording.units,
                has_any_data=recording.has_any_data,
                channel_variance=recording.channel_variance,
                annotations=[
                    Annotation(
                        caption_type="llm",
                        text=llm_text,
                        channel_idxs=active_idxs,
                    )
                ],
            )
            caption_result.rows.append(out_rec)

        n_new += 1
        print(f"[{n_new}] {row_id}")
        print(f"  tokens in/out: {response.input_tokens}/{response.output_tokens}")
        print(f"  llm ({len(llm_text)} chars): {llm_text[:200]}...")
        print()

    if args.write_shards and caption_result.rows:
        from exporters.lean import write_caption_result

        write_caption_result(caption_result, args.out)
        print(f"Wrote feather shards to {args.out}")

    if args.dry_run:
        print(f"Dry run: would process up to {len(rows)} rows.")
    else:
        print(f"Appended {n_new} captions to {jsonl_path}")


if __name__ == "__main__":
    main()
