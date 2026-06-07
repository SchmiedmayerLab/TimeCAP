#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
"""ClinicalSignalAuditor — iterates a dataset and counts per-signal fire and flag rates."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from extractors import ChannelConfig
from clinical._base import ClinicalSynthesizer
from timef.schema import Recording

FLAG_SUFFIX = ":flag"


@dataclass
class SignalStats:
    name: str
    total: int = 0
    fired: int = 0
    flagged: int = 0
    participants: set[str] = field(default_factory=set)
    flagged_participants: set[str] = field(default_factory=set)
    examples: list[str] = field(default_factory=list)
    flagged_examples: list[str] = field(default_factory=list)

    @property
    def pct_windows(self) -> float:
        return 100.0 * self.fired / self.total if self.total else 0.0

    @property
    def pct_flagged_windows(self) -> float:
        return 100.0 * self.flagged / self.total if self.total else 0.0

    @property
    def n_participants(self) -> int:
        return len(self.participants)

    @property
    def n_flagged_participants(self) -> int:
        return len(self.flagged_participants)


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
        self.all_participants: set[str] = set()

    def run(self, recordings: Iterable[Recording], desc: str = "Auditing") -> list[SignalStats]:
        for rec in tqdm(recordings, desc=desc):
            self.all_participants.add(rec.user_id)
            for stats, (_, synth) in zip(self.stats, self.synthesizers):
                stats.total += 1
                annotations = synth.synthesize(rec, self.config)
                if annotations:
                    stats.fired += 1
                    stats.participants.add(rec.user_id)
                    is_flag = any(str(ann.label or "").endswith(FLAG_SUFFIX) for ann in annotations)
                    if is_flag:
                        stats.flagged += 1
                        stats.flagged_participants.add(rec.user_id)
                        if len(stats.flagged_examples) < self.max_examples:
                            stats.flagged_examples.append(annotations[0].text)
                    if len(stats.examples) < self.max_examples:
                        stats.examples.append(annotations[0].text)
        return self.stats

    def never_fired_participants(self, signal_name: str) -> set[str]:
        """Users who never triggered a signal — for absence-based flags like sustained_bout."""
        for stats in self.stats:
            if stats.name == signal_name:
                return self.all_participants - stats.participants
        return set()

    # ── printing ──────────────────────────────────────────────────────────────

    def print_table(self, n_participants_total: int | None = None) -> None:
        n_part = n_participants_total or len(self.all_participants)
        col = [52, 16, 10, 16, 10, 14, 10]
        header = (
            f"{'Signal':<{col[0]}} {'Fired (win)':<{col[1]}} {'%Win':>{col[2]}} "
            f"{'Flagged (win)':<{col[3]}} {'%Flag':>{col[4]}} "
            f"{'Flag parts':>{col[5]}} {'%FPart':>{col[6]}}"
        )
        sep = "-" * (sum(col) + 6)
        print(sep)
        print(header)
        print(sep)
        for s in self.stats:
            pct_fpart = f"{100.0 * s.n_flagged_participants / n_part:.1f}%" if n_part else "?"
            print(
                f"{s.name:<{col[0]}} "
                f"{s.fired:>6} / {s.total:<6} "
                f"{s.pct_windows:>{col[2]}.1f}% "
                f"{s.flagged:>6} / {s.total:<6} "
                f"{s.pct_flagged_windows:>{col[4]}.1f}% "
                f"{s.n_flagged_participants:>6} / {n_part!s:<6} "
                f"{pct_fpart:>{col[6]}}"
            )
        print(sep)

    def print_label_association(
        self,
        labels_df,
        label_cols: list[str],
        absence_signal: str | None = "signal_17_sustained_bout",
    ) -> None:
        """Sensitivity / PPV / specificity for each signal × clinical label.

        Per-user:
          flagged_users = ever flagged by signal (or never fired, for absence_signal)
          TP = flagged & label_true
          FP = flagged & label_false
          FN = not_flagged & label_true
          TN = not_flagged & label_false
        """
        seen = self.all_participants
        df = labels_df[labels_df["user_id"].isin(seen)].drop_duplicates("user_id").set_index("user_id")
        available = [c for c in label_cols if c in df.columns]
        if not available:
            print("No matching label columns found.")
            return

        w = 52
        print(f"\n{'Signal':<{w}} {'Label':<34} {'Sens':>6} {'PPV':>6} {'Spec':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
        print("-" * 130)

        for stats in self.stats:
            flagged = (
                self.never_fired_participants(stats.name)
                if stats.name == absence_signal
                else stats.flagged_participants
            )
            not_flagged = seen - flagged

            for col in available:
                col_data = df[col].dropna()
                pos_vals = {"1", "1.0", "True", "true", "yes", "YES"}
                label_true = set(col_data[col_data.astype(str).isin(pos_vals)].index) & seen
                label_false = seen - label_true

                tp = len(flagged & label_true)
                fp = len(flagged & label_false)
                fn = len(not_flagged & label_true)
                tn = len(not_flagged & label_false)

                sens = f"{tp/(tp+fn):.2f}" if (tp + fn) > 0 else "  —  "
                ppv  = f"{tp/(tp+fp):.2f}" if (tp + fp) > 0 else "  —  "
                spec = f"{tn/(tn+fp):.2f}" if (tn + fp) > 0 else "  —  "

                print(f"{stats.name:<{w}} {col[:34]:<34} {sens:>6} {ppv:>6} {spec:>6} {tp:>4} {fp:>4} {fn:>4} {tn:>4}")
        print()

    def print_examples(self) -> None:
        for s in self.stats:
            if s.flagged_examples:
                print(f"\n--- {s.name} (flagged examples) ---")
                for ex in s.flagged_examples:
                    print(f"  ⚑ {ex}")
            elif s.examples:
                print(f"\n--- {s.name} (examples, informational) ---")
                for ex in s.examples:
                    print(f"  • {ex}")

    def save_csv(self, path: Path, n_participants_total: int | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        n_part = n_participants_total or len(self.all_participants)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "signal",
                "windows_fired", "windows_total", "pct_windows",
                "windows_flagged", "pct_flagged_windows",
                "participants_fired", "participants_flagged", "participants_total",
                "pct_participants_flagged",
                "example_1", "example_2", "example_3",
                "flagged_example_1", "flagged_example_2", "flagged_example_3",
            ])
            for s in self.stats:
                writer.writerow([
                    s.name,
                    s.fired, s.total, f"{s.pct_windows:.2f}",
                    s.flagged, f"{s.pct_flagged_windows:.2f}",
                    s.n_participants, s.n_flagged_participants, n_part,
                    f"{100.0 * s.n_flagged_participants / n_part:.2f}" if n_part else "",
                    *s.examples[:3],
                    *s.flagged_examples[:3],
                ])
        print(f"Saved to {path}")
