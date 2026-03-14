from __future__ import annotations

import abc

from timef.schema import SignalSpec, Signal


class Transformer(abc.ABC):
    @abc.abstractmethod
    def get_signal_specs(self) -> list[SignalSpec]: ...

    @abc.abstractmethod
    def transform_row(self, row: dict) -> list[Signal]: ...
