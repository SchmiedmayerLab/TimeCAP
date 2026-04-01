#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

try:
    from lark.exceptions import UnexpectedCharacters, UnexpectedToken
    from pint.errors import UndefinedUnitError
    from ucumvert import PintUcumRegistry
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    PintUcumRegistry = None  # type: ignore[assignment]
    UndefinedUnitError = ValueError  # type: ignore[assignment]
    UnexpectedCharacters = ValueError  # type: ignore[assignment]
    UnexpectedToken = ValueError  # type: ignore[assignment]

if TYPE_CHECKING:
    from pint import Unit

_registry: PintUcumRegistry | None = None
_valid_cache: dict[str, bool] = {}


def get_unit_registry() -> PintUcumRegistry:
    global _registry
    if PintUcumRegistry is None:
        raise RuntimeError("UCUM validation requires the optional 'pint', 'ucumvert', and 'lark' packages.")
    if _registry is None:
        _registry = PintUcumRegistry()
    return _registry


def validate_ucum(unit_code: str) -> str:
    cached = _valid_cache.get(unit_code)
    if cached is True:
        return unit_code
    if cached is False:
        raise ValueError(f"Invalid UCUM unit code '{unit_code}'")

    if PintUcumRegistry is None:
        _valid_cache[unit_code] = True
        return unit_code

    try:
        get_unit_registry().from_ucum(unit_code)
        _valid_cache[unit_code] = True
    except (UndefinedUnitError, ValueError, KeyError, UnexpectedCharacters, UnexpectedToken) as exc:
        _valid_cache[unit_code] = False
        raise ValueError(f"Invalid UCUM unit code '{unit_code}': {exc}") from exc
    return unit_code


@lru_cache(maxsize=256)
def ucum_to_pint(unit_code: str) -> Unit:
    return get_unit_registry().from_ucum(unit_code)
