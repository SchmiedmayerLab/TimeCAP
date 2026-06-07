#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""ClinicalSignalAuditor — iterates a dataset and counts per-signal fire rates."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from extractors import ChannelConfig
from clinical._base import ClinicalSynthesizer
from timef.schema import Recording


@dataclass
class SignalStats:
    name: str
    total: int = 0
    fired: int = 0
    participants: set[str] = field(default_factory=set)
    examples: list[str] = field(default_factory=list)

    @property
    def pct_windows(self) -> float:
        return 100.0 * self.fired / self.total if self.total else 0.0

    @property
    def n_participants(self) -> int:
        return len(self.participants)


class ClinicalSignalAuditor:
    def __init__(
        self,
        synthesizers: list[tuple[str, ClinicalSynthesizer]],
        config: ChannelConfig,
        max_examples: int = 3,
    ):
        self.synthesizers = synthesizers
        self.config = config
        self.max_examples = max_examples
        self.stats: list[SignalStats] = [SignalStats(name=name) for name, _ in synthesizers]

    def run(self, recordings: Iterable[Recording], desc: str = "Auditing") -> list[SignalStats]:
        for rec in tqdm(recordings, desc=desc):
            for stats, (_, synth) in zip(self.stats, self.synthesizers):
                stats.total += 1
                annotations = synth.synthesize(rec, self.config)
                if annotations:
                    stats.fired += 1
                    stats.participants.add(rec.user_id)
                    if len(stats.examples) < self.max_examples:
                        stats.examples.append(annotations[0].text)
        return self.stats

    def print_table(self, n_participants_total: int | None = None) -> None:
        col = [60, 18, 10, 14, 16]
        header = (
            f"{'Signal':<{col[0]}} {'Windows fired':<{col[1]}} {'%Win':>{col[2]}} "
            f"{'Participants':>{col[3]}} {'%Part':>{col[4]}}"
        )
        sep = "-" * sum(col)
        print(sep)
        print(header)
        print(sep)
        for s in self.stats:
            n_part = n_participants_total or "?"
            pct_part = (
                f"{100.0 * s.n_participants / n_participants_total:.1f}%"
                if n_participants_total else "?"
            )
            print(
                f"{s.name:<{col[0]}} "
                f"{s.fired:>6} / {s.total:<8} "
                f"{s.pct_windows:>{col[2]}.1f}% "
                f"{s.n_participants:>6} / {n_part!s:<6} "
                f"{pct_part:>{col[4]}}"
            )
        print(sep)

    def print_examples(self) -> None:
        for s in self.stats:
            if s.examples:
                print(f"\n--- {s.name} (examples) ---")
                for ex in s.examples:
                    print(f"  • {ex}")

    def save_csv(self, path: Path, n_participants_total: int | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "signal", "windows_fired", "windows_total", "pct_windows",
                "participants_fired", "participants_total", "pct_participants",
                "example_1", "example_2", "example_3",
            ])
            for s in self.stats:
                writer.writerow([
                    s.name,
                    s.fired, s.total, f"{s.pct_windows:.2f}",
                    s.n_participants, n_participants_total or "",
                    f"{100.0 * s.n_participants / n_participants_total:.2f}" if n_participants_total else "",
                    *s.examples[:3],
                ])
        print(f"Saved to {path}")
