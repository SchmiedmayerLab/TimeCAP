#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the TimeCap open-source project.
#
# SPDX-License-Identifier: MIT
#
from models.base import BaseModel, ModelResponse
from models.client import ClientConfig, ClientModel

__all__ = [
    "BaseModel",
    "ClientConfig",
    "ClientModel",
    "LocalConfig",
    "LocalModel",
    "ModelResponse",
]


def __getattr__(name: str):
    if name in ("LocalConfig", "LocalModel"):
        from models.local import LocalConfig, LocalModel

        return LocalConfig if name == "LocalConfig" else LocalModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
