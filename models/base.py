#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
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
    """Abstract model backend."""

    def __init__(self, channel_config: ChannelConfig | None = None):
        self.channel_config = channel_config

    @abc.abstractmethod
    def generate(self, sample: dict, recording: Recording) -> ModelResponse:
        """Run inference on one sample.

        ``recording`` carries raw per-channel values for plot rendering.
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
