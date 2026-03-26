#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import textwrap
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

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

    # Collect per-channel captions grouped by type, e.g. {"heart_rate": {"statistical": ["mean is..."]}}
    channel_captions: dict[str, dict[str, list[str]]] = {ch: {} for ch in channel_names}
    for ann in annotations:
        if not ann.answer:
            continue
        ann_type = ann.spec_id.split(":")[-1] if ":" in ann.spec_id else ann.spec_id
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
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return None
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    from mhc.dataset import MHCDataset
    from mhc.transformer import MHCTransformer
    from mhc.constants import MHC_CHANNEL_CONFIG
    from extractors.statistical import StatisticalExtractor
    from extractors.structural import StructuralExtractor
    from annotator import Annotator
    from captionizer import Captionizer

    dataset = MHCDataset(min_wear_pct=90.0)
    annotator = Annotator([StatisticalExtractor(MHC_CHANNEL_CONFIG), StructuralExtractor(MHC_CHANNEL_CONFIG)])
    captionizer = Captionizer(dataset, MHCTransformer(), annotator)
    result = captionizer.run(max_rows=1)
    for signals, samples, annotations in result.iter_rows():
        plot_row(signals, samples, annotations, save_path="sample_plot.png")
        print("Saved sample_plot.png")
