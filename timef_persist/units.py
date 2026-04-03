from __future__ import annotations


def validate_ucum(unit_code: str) -> str:
    if not isinstance(unit_code, str) or not unit_code.strip():
        raise ValueError("Unit code must be a non-empty string")
    return unit_code
