"""Compatibility shim for legacy timef.schema imports.

Prefer importing runtime types from `runtime.types`.
"""

from runtime.types import Annotation, CaptionResult, RuntimeRow, SignalView

__all__ = [
    "Annotation",
    "CaptionResult",
    "RuntimeRow",
    "SignalView",
]
