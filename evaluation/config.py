#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Evaluation configuration dataclasses.

All configs are frozen so they can be serialised to JSON without mutation risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from models.client import ClientConfig
from models.local import LocalConfig


@dataclass(frozen=True)
class ModelConfig:
    """Identifies one model to evaluate.

    Exactly one of client_config or local_config must be set.

    Attributes:
        name:          Display name used in result file paths (e.g. "gpt-4o").
        client_config: Config for an OpenAI-compatible API endpoint.
        local_config:  Config for a local HuggingFace transformers model.
    """
    name: str
    client_config: ClientConfig | None = None
    local_config: LocalConfig | None = None

    def __post_init__(self) -> None:
        if self.client_config is None and self.local_config is None:
            raise ValueError(f"ModelConfig '{self.name}': set either client_config or local_config.")
        if self.client_config is not None and self.local_config is not None:
            raise ValueError(f"ModelConfig '{self.name}': set only one of client_config or local_config.")


@dataclass(frozen=True)
class EvalConfig:
    """Top-level evaluation run configuration.

    Attributes:
        task_name:     Name of the EvalTask to run (used for result paths).
        models:        Models to evaluate (run sequentially, results saved per model).
        max_samples:   Cap on samples evaluated per model (None = all).
        split:         Dataset split to evaluate on.
        results_dir:   Root directory for result files.
        seed:          Global random seed passed to the dataset.
    """
    task_name: str
    models: tuple[ModelConfig, ...]
    max_samples: int | None = None
    split: Literal["train", "validation", "test"] = "test"
    results_dir: Path = Path("results")
    seed: int = 42
