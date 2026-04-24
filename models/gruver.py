#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""Digit-level serializer for 1D arrays (Gruver et al. 2023, LLMTime).

Copied from OpenTSLM's ``evaluation/baseline/gruver_llmtime_tokenizer.py``
so text-only LLM evaluation stays byte-identical with their baseline. The
PyPI ``opentslm`` package ships only the library code, not this helper.
Original: https://github.com/ngruver/llmtime/blob/main/data/serialize.py
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np


@dataclass
class SerializerSettings:
    base: int = 10
    prec: int = 3
    signed: bool = True
    fixed_length: bool = False
    max_val: float = 1e7
    time_sep: str = " ,"
    bit_sep: str = " "
    plus_sign: str = ""
    minus_sign: str = " -"
    half_bin_correction: bool = True
    decimal_point: str = ""
    missing_str: str = " Nan"


def _vec_num2repr(val: np.ndarray, base: int, prec: int, max_val: float):
    base_f = float(base)
    sign = np.where(val >= 0, 1, -1)
    mag = np.abs(val)
    max_bit_pos = int(np.ceil(np.log(max_val) / np.log(base_f)).item())

    before = []
    rem = mag.copy()
    for i in range(max_bit_pos):
        power = base_f ** (max_bit_pos - i - 1)
        digit = (rem / power).astype(int)
        before.append(digit)
        rem = rem - digit * power
    before = np.stack(before, axis=-1)

    if prec > 0:
        after = []
        for i in range(prec):
            power = base_f ** (-(i + 1))
            digit = (rem / power).astype(int)
            after.append(digit)
            rem = rem - digit * power
        after = np.stack(after, axis=-1)
        digits = np.concatenate([before, after], axis=-1)
    else:
        digits = before
    return sign, digits


def serialize_arr(arr: np.ndarray, settings: SerializerSettings) -> str:
    clean = np.where(np.isnan(arr), 0.0, arr)
    assert np.all(np.abs(clean) <= settings.max_val), (
        f"Values must be within ±{settings.max_val}"
    )

    to_repr = partial(
        _vec_num2repr, base=settings.base, prec=settings.prec, max_val=settings.max_val
    )
    sign_arr, digits_arr = to_repr(clean)
    is_nan = np.isnan(arr)

    def format_digits(digits):
        if not settings.fixed_length:
            nz = np.where(digits != 0)[0]
            digits = digits[nz[0]:] if nz.size > 0 else np.array([0], dtype=int)
        if settings.decimal_point and settings.prec > 0:
            point_idx = len(digits) - settings.prec
            digits = np.concatenate([digits[:point_idx], [-1], digits[point_idx:]])
        return settings.bit_sep.join(str(d) for d in digits if d != -1)

    parts = []
    for s, digs, missing in zip(sign_arr, digits_arr, is_nan):
        if missing:
            parts.append(settings.missing_str)
        else:
            sign_str = settings.plus_sign if s > 0 else settings.minus_sign
            parts.append(sign_str + format_digits(digs))
    return settings.time_sep.join(parts) + settings.time_sep


_GPT_SETTINGS = SerializerSettings(
    base=10, prec=3, signed=True, time_sep=", ", bit_sep=" ", minus_sign="-"
)
_LLAMA_SETTINGS = SerializerSettings(
    base=10, prec=3, signed=True, half_bin_correction=True
)


def gpt_formatter(arr: np.ndarray) -> str:
    return serialize_arr(arr, _GPT_SETTINGS)


def llama_formatter(arr: np.ndarray) -> str:
    return serialize_arr(arr, _LLAMA_SETTINGS)


def pick_formatter(model_name: str):
    """Return the formatter OpenTSLM's CommonEvaluator would pick for ``model_name``."""
    lower = model_name.lower()
    if lower.startswith("openai-") or "gpt" in lower:
        return gpt_formatter
    return llama_formatter
