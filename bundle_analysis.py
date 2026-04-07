#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from random import Random

import numpy as np
from datasets import load_from_disk

from extractors.bundle import BundleExtractor
from mhc.constants import CHANNEL_NAMES, DATASET_DIR, MHC_CHANNEL_CONFIG
from mhc.dataset import MHCDataset
from mhc.transformer import MHCTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze multi-signal bundle co-occurrence on the MHC dataset.")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--max-rows", type=int, default=15000, help="Rows to scan for window-level bundle extraction.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def format_metric(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def format_window(window: tuple[int, int]) -> str:
    start, end = window
    return f"{start // 60:02d}:{start % 60:02d}-{end // 60:02d}:{end % 60:02d}"


def summarize_quantiles(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "n/a"
    stats = {q: np.percentile(arr, q) for q in (25, 50, 75, 90)}
    return ", ".join(f"p{q}={format_metric(v)}" for q, v in stats.items())


def row_level_cooccurrence(dataset_dir: str, top_k: int) -> None:
    ds = load_from_disk(dataset_dir)
    has_any = np.asarray(ds["has_any_data"], dtype=bool)
    counts = has_any.sum(axis=0)

    print(f"Rows in dataset: {has_any.shape[0]}")
    print("\nSignal coverage by row:")
    for index, name in enumerate(CHANNEL_NAMES):
        pct = counts[index] / has_any.shape[0] * 100
        print(f"- {name}: {int(counts[index])} rows ({pct:.1f}%)")

    pairs = []
    for i in range(has_any.shape[1]):
        for j in range(i + 1, has_any.shape[1]):
            count = int(np.logical_and(has_any[:, i], has_any[:, j]).sum())
            if count:
                pairs.append((count, CHANNEL_NAMES[i], CHANNEL_NAMES[j]))

    print(f"\nTop {top_k} row-level co-occurring signal pairs:")
    for count, left, right in sorted(pairs, reverse=True)[:top_k]:
        print(f"- {count}: {left} + {right}")


def bundle_level_analysis(dataset_dir: str, max_rows: int, top_k: int, seed: int) -> None:
    dataset = MHCDataset(dataset_dir=dataset_dir)
    transformer = MHCTransformer()
    extractor = BundleExtractor(MHC_CHANNEL_CONFIG)

    limit = min(max_rows, len(dataset))
    rng = Random(seed)
    sampled_indices = rng.sample(range(len(dataset)), limit) if limit < len(dataset) else list(range(len(dataset)))
    label_counts: Counter[str] = Counter()
    label_examples: dict[str, list[tuple[str, tuple[int, int], str]]] = defaultdict(list)
    label_durations: defaultdict[str, list[float]] = defaultdict(list)
    workout_metrics: defaultdict[str, list[dict[str, float]]] = defaultdict(list)

    for row_index in sampled_indices:
        row = transformer.transform_row(dataset[row_index])
        annotations = extractor.extract(row)
        for annotation in annotations:
            label = str(annotation.label)
            label_counts[label] += 1
            if annotation.window is not None:
                start, end = annotation.window
                label_durations[label].append(float(end - start))
            if len(label_examples[label]) < 3:
                label_examples[label].append((row.row_id, annotation.window or (0, 0), annotation.text))

            if label.startswith("workout:") and annotation.window is not None:
                idx = {name: i for i, name in enumerate(row.channel_names)}
                start, end = annotation.window
                hr = row.values[idx["hk_watch:HKQuantityTypeIdentifierHeartRate"]][start:end]
                steps = (
                    np.nan_to_num(row.values[idx["hk_watch:HKQuantityTypeIdentifierStepCount"]][start:end], nan=0.0)
                    + np.nan_to_num(row.values[idx["hk_iphone:HKQuantityTypeIdentifierStepCount"]][start:end], nan=0.0)
                )
                energy = np.nan_to_num(row.values[idx["hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned"]][start:end], nan=0.0)
                distance = (
                    np.nan_to_num(row.values[idx["hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning"]][start:end], nan=0.0)
                    + np.nan_to_num(row.values[idx["hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning"]][start:end], nan=0.0)
                )
                workout_metrics[label].append(
                    {
                        "minutes": float(end - start),
                        "hr_mean": float(np.nanmean(hr)) if np.isfinite(hr).any() else np.nan,
                        "steps": float(steps.sum()),
                        "energy": float(energy.sum()),
                        "distance": float(distance.sum()),
                    }
                )

    print(f"\nBundle analysis rows scanned: {limit} random rows (seed={seed})")
    print("\nBundle counts:")
    for label, count in label_counts.most_common():
        duration_summary = summarize_quantiles(label_durations[label])
        print(f"- {label}: {count} hits, duration {duration_summary}")

    print(f"\nTop {top_k} bundle examples:")
    shown = 0
    for label, _ in label_counts.most_common():
        for row_id, window, text in label_examples[label]:
            print(f"- {label} | {row_id} | {format_window(window)} | {text}")
            shown += 1
            if shown >= top_k:
                break
        if shown >= top_k:
            break

    print("\nWorkout bundle summaries:")
    for label, metrics in sorted(workout_metrics.items(), key=lambda item: len(item[1]), reverse=True):
        minutes = [item["minutes"] for item in metrics]
        hr_mean = [item["hr_mean"] for item in metrics]
        steps = [item["steps"] for item in metrics]
        energy = [item["energy"] for item in metrics]
        distance = [item["distance"] for item in metrics]
        print(f"- {label}: {len(metrics)} hits")
        print(f"  minutes {summarize_quantiles(minutes)}")
        print(f"  hr_mean {summarize_quantiles(hr_mean)}")
        print(f"  steps {summarize_quantiles(steps)}")
        print(f"  distance {summarize_quantiles(distance)}")
        print(f"  energy {summarize_quantiles(energy)}")


def main() -> None:
    args = parse_args()
    row_level_cooccurrence(args.dataset_dir, args.top_k)
    bundle_level_analysis(args.dataset_dir, args.max_rows, args.top_k, args.seed)


if __name__ == "__main__":
    main()
