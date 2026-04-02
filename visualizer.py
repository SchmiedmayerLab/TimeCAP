#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import argparse
import textwrap
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from extractors import ChannelConfig
from timef.schema import Annotation, Sample, Signal


def plot_row(
    signals: list[Signal],
    samples: list[Sample],
    annotations: list[Annotation],
    save_path: str | None = None,
    n_cols: int = 3,
) -> plt.Figure | None:
    channel_names = [s.name for s in signals]
    n_channels = len(channel_names)

    data = np.stack([s.data for s in signals])
    time_minutes = np.arange(data.shape[1])

    # Build signal_id -> channel_name
    sig_id_to_name = {s.id: s.name for s in signals}

    # Build sample_id -> list of channel names
    sample_channels: dict[str, list[str]] = {}
    for sample in samples:
        names = [
            sig_id_to_name[ref.signal_id]
            for ref in sample.signals
            if ref.signal_id in sig_id_to_name
        ]
        if names:
            sample_channels[sample.id] = names

    # Split annotations: semantic captions are global, others are per-channel
    semantic_captions: list[str] = []
    channel_captions: dict[str, dict[str, list[str]]] = {ch: {} for ch in channel_names}
    for ann in annotations:
        if not ann.answer:
            continue
        ann_type = ann.spec_id.split(":")[-1] if ":" in ann.spec_id else ann.spec_id
        if ann_type == "semantic":
            semantic_captions.append(ann.answer)
        else:
            for sample_ref in ann.samples:
                for ch in sample_channels.get(sample_ref.sample_id, []):
                    if ch in channel_captions:
                        channel_captions[ch].setdefault(ann_type, []).append(ann.answer)

    # Derive per-channel nonwear regions from NaN
    def _nan_regions(arr: np.ndarray, min_length: int = 30) -> list[tuple[int, int]]:
        regions = []
        in_region = False
        for i, val in enumerate(np.isnan(arr)):
            if val and not in_region:
                start = i
                in_region = True
            elif not val and in_region:
                if i - start >= min_length:
                    regions.append((start, i))
                in_region = False
        if in_region and len(arr) - start >= min_length:
            regions.append((start, len(arr)))
        return regions

    n_rows = ceil(n_channels / n_cols)
    fig, axes_2d = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), sharex="col")
    axes = axes_2d.flatten()

    for i, (ax, name) in enumerate(zip(axes[:n_channels], channel_names)):
        valid = ~np.isnan(data[i])
        ax.plot(time_minutes[valid], data[i][valid], linewidth=0.8, color="steelblue")
        for t0, t1 in _nan_regions(data[i]):
            ax.axvspan(t0, t1, color="red", alpha=0.05)
        ax.set_ylabel(name, fontsize=6, rotation=90, ha="right", va="center")
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", labelsize=6, labelbottom=True)
        ax.set_xlim(0, data.shape[1])

    for col in range(n_cols):
        axes[col].set_xticks(range(0, data.shape[1] + 1, 200))

    for ax in axes[n_channels:]:
        ax.set_visible(False)

    bottom_row_indices = {
        (n_channels - 1 - col) // n_cols * n_cols + col
        for col in range(min(n_cols, n_channels))
    }

    for i in range(n_channels):
        ax = axes[i]
        caps = channel_captions.get(channel_names[i], {})
        parts = []
        for ann_type in ("statistical", "structural", "semantic"):
            if ann_type in caps:
                parts.append(textwrap.fill(" ".join(caps[ann_type]), width=80))
        if parts:
            caption_text = "\n".join("\n".join(parts).splitlines()[:10])
            ax.set_xlabel(caption_text, fontsize=6, labelpad=4)
        elif i in bottom_row_indices:
            ax.set_xlabel("Time (minutes)")

    meta = signals[0].metadata if signals else {}
    title = f"{meta.get('user_id', '')}  |  {meta.get('date', '')}"
    fig.suptitle(title, fontsize=10, y=1.0)

    if semantic_captions:
        sem_text = textwrap.fill("  ".join(semantic_captions), width=140)
        n_lines = sem_text.count("\n") + 1
        line_height = (7 / 72) / fig.get_size_inches()[1]
        fig.text(0.5, 0.98, sem_text, ha="center", va="top", fontsize=7, style="italic")
        fig.tight_layout(rect=[0, 0, 1, 0.96 - line_height * max(n_lines - 1, 0)])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return None
    else:
        plt.show()
    return fig


def plot_detector_view(
    signals: list[Signal],
    config: ChannelConfig,
    save_path: str | None = None,
    n_cols: int = 3,
) -> plt.Figure | None:
    channel_names = [s.name for s in signals]
    n_channels = len(channel_names)
    if n_channels == 0:
        raise ValueError("signals must not be empty")

    data = np.stack([s.data for s in signals])
    time_minutes = np.arange(data.shape[1])

    n_rows = ceil(n_channels / n_cols)
    fig, axes_2d = plt.subplots(n_rows, n_cols, figsize=(20, 4.6 * n_rows), sharex="col")
    axes = np.atleast_1d(axes_2d).flatten()

    legend_handles: dict[str, object] = {
        "signal": Line2D([0], [0], color="steelblue", linewidth=1.0, label="signal"),
        "trend:increasing": Patch(facecolor="#4daf4a", edgecolor="none", alpha=0.18, label="trend (increasing)"),
        "trend:decreasing": Patch(facecolor="#ff7f00", edgecolor="none", alpha=0.18, label="trend (decreasing)"),
        "plateau": Patch(facecolor="#377eb8", edgecolor="none", alpha=0.16, label="plateau"),
        "spike": Line2D([0], [0], marker="^", linestyle="None", color="#2ca02c", markersize=6, label="spike"),
        "drop": Line2D([0], [0], marker="v", linestyle="None", color="#d62728", markersize=6, label="drop"),
        "gap": Patch(facecolor="#d62728", edgecolor="none", alpha=0.10, label="gap"),
    }

    for i, (ax, signal) in enumerate(zip(axes[:n_channels], signals)):
        series = data[i]
        valid = ~np.isnan(series)
        ax.plot(time_minutes[valid], series[valid], linewidth=0.8, color="steelblue")

        detector_summaries: list[str] = []
        for detector in config.detectors.get(signal.name, []):
            detector_name = detector.__class__.__name__
            results = detector.detect(signal.data)
            if not results:
                detector_summaries.append(f"{detector_name}: none")
                continue

            rendered = []
            for result in results:
                if result.event_type == "trend":
                    direction = result.direction or "trend"
                    color = "#4daf4a" if direction == "increasing" else "#ff7f00"
                    ax.axvspan(result.start_minute, result.end_minute, color=color, alpha=0.18)
                    rendered.append(f"{direction} {result.start_minute}-{result.end_minute}")
                elif result.event_type == "plateau":
                    ax.axvspan(result.start_minute, result.end_minute, color="#377eb8", alpha=0.16)
                    rendered.append(f"plateau {result.start_minute}-{result.end_minute}")
                elif result.event_type == "spike":
                    minute = int(result.spike_minute)
                    value = signal.data[minute]
                    if not np.isnan(value):
                        ax.scatter(minute, value, color="#2ca02c", marker="^", s=28, zorder=3)
                    rendered.append(f"spike @{minute}")
                elif result.event_type == "drop":
                    minute = int(result.spike_minute)
                    value = signal.data[minute]
                    if not np.isnan(value):
                        ax.scatter(minute, value, color="#d62728", marker="v", s=28, zorder=3)
                    rendered.append(f"drop @{minute}")
                elif result.event_type == "gap":
                    ax.axvspan(result.start_minute, result.end_minute, color="#d62728", alpha=0.10)
                    rendered.append(f"gap {result.start_minute}-{result.end_minute}")
                else:
                    rendered.append(result.event_type)

            detector_summaries.append(f"{detector_name}: {', '.join(rendered)}")

        display_name = config.meta.get(signal.name, (signal.name, "", 0))[0]
        ax.set_title(display_name, fontsize=8)
        ax.set_ylabel(signal.name, fontsize=6, rotation=90, ha="right", va="center")
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", labelsize=6, labelbottom=True)
        ax.set_xlim(0, len(series))

        summary_text = "\n".join(textwrap.fill(line, width=56) for line in detector_summaries)
        if summary_text:
            ax.text(
                0.01,
                0.99,
                summary_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=5.5,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
            )

    for col in range(n_cols):
        axes[col].set_xticks(range(0, data.shape[1] + 1, 200))

    for ax in axes[n_channels:]:
        ax.set_visible(False)

    for ax in axes[:n_channels]:
        ax.set_xlabel("Time (minutes)")

    meta = signals[0].metadata if signals else {}
    title = f"{meta.get('user_id', '')}  |  {meta.get('date', '')}  |  detector debug view"
    fig.suptitle(title, fontsize=10)
    fig.legend(
        handles=list(legend_handles.values()),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=6,
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return None

    plt.show()
    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize caption or detector-debug plots for one dataset row.")
    parser.add_argument("--mode", choices=("captions", "detectors"), default="captions")
    parser.add_argument("--row-index", type=int, default=0, help="Dataset row index to visualize.")
    parser.add_argument("--save-path", type=str, default=None, help="Optional output image path.")
    parser.add_argument("--min-wear-pct", type=float, default=90.0, help="Minimum wear percentage filter.")
    return parser.parse_args()


if __name__ == "__main__":
    from mhc.dataset import MHCDataset
    from mhc.transformer import MHCTransformer
    from mhc.constants import MHC_CHANNEL_CONFIG
    from extractors.statistical import StatisticalExtractor
    from extractors.structural import StructuralExtractor
    from annotator import Annotator
    from captionizer import Captionizer

    args = _parse_args()

    dataset = MHCDataset(min_wear_pct=args.min_wear_pct)
    if args.row_index < 0 or args.row_index >= len(dataset):
        raise IndexError(f"row-index must be between 0 and {len(dataset) - 1}, got {args.row_index}")

    row = dataset[args.row_index]
    signals = MHCTransformer().transform_row(row)

    if args.mode == "detectors":
        save_path = args.save_path or f"detector_debug_row_{args.row_index}.png"
        plot_detector_view(signals, MHC_CHANNEL_CONFIG, save_path=save_path)
        print(f"Saved {save_path}")
    else:
        annotator = Annotator([StatisticalExtractor(MHC_CHANNEL_CONFIG), StructuralExtractor(MHC_CHANNEL_CONFIG)])
        samples, annotations = annotator.annotate(signals)
        save_path = args.save_path or "sample_plot.png"
        plot_row(signals, samples, annotations, save_path=save_path)
        print(f"Saved {save_path}")
