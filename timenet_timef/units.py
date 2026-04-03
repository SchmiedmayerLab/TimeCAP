#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations


def validate_ucum(unit_code: str) -> str:
    if not isinstance(unit_code, str) or not unit_code.strip():
        raise ValueError("Unit code must be a non-empty string")
    return unit_code
