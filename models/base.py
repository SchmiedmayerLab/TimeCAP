#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import abc
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from extractors import ChannelConfig
from timef.schema import Annotation, Signal


@dataclass
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int


class BaseModel(abc.ABC):
    """Abstract base for model backends. Renders signals, assembles context, calls the model."""

    def __init__(self, channel_config: ChannelConfig):
        self.channel_config = channel_config

    def process(
        self,
        prompt: str,
        signals: list[Signal],
        annotations: list[Annotation] | None = None,
        multi_channel: bool = False,
    ) -> ModelResponse:
        """Render signals as plots, assemble with optional annotations, call model."""
        caption = self._format_annotations(annotations) if annotations else None
        return self._call(
            prompt=prompt, signals=signals, caption=caption, multi_channel=multi_channel,
        )

    @abc.abstractmethod
    def _call(
        self,
        prompt: str,
        signals: list[Signal],
        caption: str | None,
        multi_channel: bool,
    ) -> ModelResponse:
        """Subclasses render signals in their native format and call the model."""
        ...

    def _plot_signal(self, signal: Signal) -> plt.Figure:
        """Plot a single signal. Subclasses convert the figure to their native image format."""
        meta = self.channel_config.meta.get(signal.name)
        title = f"{meta[0]} ({meta[1]})" if meta else signal.name

        data = signal.data
        time = np.arange(len(data))
        valid = ~np.isnan(data)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(time[valid], data[valid], linewidth=1, color="steelblue")
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, len(data))
        ax.set_xlabel("Minutes")
        fig.tight_layout()
        return fig

    def _format_annotations(self, annotations: list[Annotation]) -> str:
        """Concatenate all annotation answers."""
        return "\n".join(ann.answer for ann in annotations if ann.answer)
