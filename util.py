#
# This source file is part of the TimeCAP open-source project
#
# SPDX-FileCopyrightText: 2026 Schmiedmayer Lab, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import zlib


def seed_from_key(key: str) -> int:
    return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF


def capitalize(text: str) -> str:
    for i, c in enumerate(text):
        if not c.isalpha():
            continue
        if c.isupper():
            return text
        # Preserve intentional lowercase-first names like iPhone, eBay, iOS.
        if i + 1 < len(text) and text[i + 1].isupper():
            return text
        return text[:i] + c.upper() + text[i + 1:]
    return text
