#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import textwrap
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

from timef.schema import Recording


def plot_row(
    row: Recording,
    save_path: str | None = None,
    n_cols: int = 3,
) -> plt.Figure | None:
    channel_names = list(row.channel_names)
    n_channels = len(channel_names)
    data = row.values
    time_minutes = np.arange(data.shape[1])

    semantic_captions = [
        annotation.text for annotation in row.annotations if annotation.caption_type == "semantic" and annotation.text
    ]
    channel_captions: dict[str, dict[str, list[str]]] = {name: {} for name in channel_names}

    for caption in row.annotations:
        if not caption.text or caption.caption_type == "semantic":
            continue
        for idx in caption.channel_idxs:
            name = channel_names[idx]
            channel_captions[name].setdefault(caption.caption_type, []).append(caption.text)

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

    fig.suptitle(f"{row.user_id}  |  {row.date}", fontsize=10, y=1.0)

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


if __name__ == "__main__":
    from mhc.dataset import MHCDataset
    from mhc.transformer import MHCTransformer
    from mhc.constants import MHC_CHANNEL_CONFIG
    from mhc.cross_channel import default_extractor
    from extractors.statistical import StatisticalExtractor
    from extractors.structural import StructuralExtractor
    from annotator import Annotator
    from captionizer import Captionizer

    dataset = MHCDataset(min_wear_pct=90.0)
    annotator = Annotator([
        StatisticalExtractor(MHC_CHANNEL_CONFIG),
        StructuralExtractor(MHC_CHANNEL_CONFIG),
        default_extractor(MHC_CHANNEL_CONFIG),
    ])
    captionizer = Captionizer(dataset, MHCTransformer(), annotator)
    result, _ = captionizer.run(max_rows=1)
    for row in result.iter_rows():
        plot_row(row, save_path="sample_plot.png")
        print("Saved sample_plot.png")
