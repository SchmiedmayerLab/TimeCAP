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
    # per-user day counts for flag-rate analysis
    user_flagged_days: dict[str, int] = field(default_factory=dict)
    user_total_days: dict[str, int] = field(default_factory=dict)

    def flag_rate(self, user_id: str) -> float:
        total = self.user_total_days.get(user_id, 0)
        return self.user_flagged_days.get(user_id, 0) / total if total else 0.0

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
                uid = rec.user_id
                stats.user_total_days[uid] = stats.user_total_days.get(uid, 0) + 1
                if annotations:
                    stats.fired += 1
                    stats.participants.add(uid)
                    is_flag = any(str(ann.label or "").endswith(FLAG_SUFFIX) for ann in annotations)
                    if is_flag:
                        stats.flagged += 1
                        stats.flagged_participants.add(uid)
                        stats.user_flagged_days[uid] = stats.user_flagged_days.get(uid, 0) + 1
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
        flag_rate_threshold: float | None = None,
    ) -> None:
        """Per-user flag-rate comparison across clinical labels.

        For each signal × label:
          - Mean flag rate (% of days flagged) for label=True vs label=False users
          - Binary sens/spec/PPV using flag_rate_threshold (default: ever-flagged)

        absence_signal: signal where absence of fires = flagged (sustained_bout).
        flag_rate_threshold: if set, user is flagged only if flag_rate > threshold (0–1).
        """
        import numpy as np

        seen = self.all_participants
        df = labels_df[labels_df["user_id"].isin(seen)].drop_duplicates("user_id").set_index("user_id")
        available = [c for c in label_cols if c in df.columns]
        if not available:
            print("No matching label columns found.")
            return

        pos_vals = {"1", "1.0", "True", "true", "yes", "YES"}
        w = 52

        print(f"\n{'Signal':<{w}} {'Label':<34} {'Rate|T':>7} {'Rate|F':>7} {'Δrate':>7} {'Sens':>6} {'PPV':>6} {'Spec':>6} {'n+':>4} {'n-':>4}")
        print("-" * 145)

        for stats in self.stats:
            is_absence = stats.name == absence_signal

            for col in available:
                col_data = df[col].dropna()
                label_true = set(col_data[col_data.astype(str).isin(pos_vals)].index) & seen
                label_false = seen - label_true
                if not label_true:
                    continue

                # Per-user flag rates
                if is_absence:
                    # absence signal: flag_rate = 1 - fire_rate
                    rates = {uid: 1.0 - stats.flag_rate(uid) for uid in seen}
                else:
                    rates = {uid: stats.flag_rate(uid) for uid in seen}

                rate_pos = [rates[u] for u in label_true if u in rates]
                rate_neg = [rates[u] for u in label_false if u in rates]
                mean_pos = float(np.mean(rate_pos)) if rate_pos else 0.0
                mean_neg = float(np.mean(rate_neg)) if rate_neg else 0.0
                delta = mean_pos - mean_neg

                # Binary classification using threshold
                if flag_rate_threshold is not None:
                    flagged = {uid for uid, r in rates.items() if r > flag_rate_threshold}
                else:
                    flagged = (
                        self.never_fired_participants(stats.name)
                        if is_absence else stats.flagged_participants
                    )
                not_flagged = seen - flagged

                tp = len(flagged & label_true)
                fp = len(flagged & label_false)
                fn = len(not_flagged & label_true)
                tn = len(not_flagged & label_false)

                sens = f"{tp/(tp+fn):.2f}" if (tp + fn) > 0 else "  — "
                ppv  = f"{tp/(tp+fp):.2f}" if (tp + fp) > 0 else "  — "
                spec = f"{tn/(tn+fp):.2f}" if (tn + fp) > 0 else "  — "

                print(
                    f"{stats.name:<{w}} {col[:34]:<34} "
                    f"{mean_pos:>7.1%} {mean_neg:>7.1%} {delta:>+7.1%} "
                    f"{sens:>6} {ppv:>6} {spec:>6} "
                    f"{len(label_true):>4} {len(label_false):>4}"
                )
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
