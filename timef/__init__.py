#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Compatibility shim for legacy timef imports.

Prefer importing runtime types from `runtime` and export helpers from
`exporters.timef_export`.
"""

from exporters.timef_export import TimeFExportConfig, export_caption_result
from runtime import Annotation, CaptionResult, RuntimeRow, SignalView

__all__ = [
    "Annotation",
    "CaptionResult",
    "RuntimeRow",
    "SignalView",
    "TimeFExportConfig",
    "export_caption_result",
]
