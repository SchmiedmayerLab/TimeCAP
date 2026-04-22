#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import zlib


def seed_from_key(key: str) -> int:
    return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF
