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
from timef.schema import Recording, SignalView


@dataclass
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int


class BaseModel(abc.ABC):
    """Abstract model backend. Set ``format_sample_str=True`` for text-only LLMs (Gruver-tokenized prompt), else False."""

    format_sample_str: bool = False

    def __init__(self, channel_config: ChannelConfig | None = None):
        self.channel_config = channel_config

    @abc.abstractmethod
    def generate(self, sample: dict, recording: Recording) -> ModelResponse:
        """Run inference on one sample.

        ``sample`` matches ``format_sample_str`` (see class docstring).
        ``recording`` carries raw per-channel values for plot rendering;
        text-only backends ignore it.
        """
        ...

    def _plot_signal(self, signal: SignalView) -> plt.Figure:
        """Plot a single signal. Subclasses convert the figure to their native image format."""
        title = signal.display_name
        if signal.unit:
            title = f"{title} ({signal.unit})"

        data = signal.data
        time = np.arange(len(data))
        valid = ~np.isnan(data)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(time[valid], data[valid], linewidth=1, color="steelblue")
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, len(data))
        ax.set_xlabel(self.channel_config.time_unit.capitalize())
        fig.tight_layout()
        return fig
