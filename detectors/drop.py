#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np
from scipy.signal import peak_widths

from detectors import DetectionResult, StructuralDetector


class DropDetector(StructuralDetector):
    """Detects genuine downward events and baseline reversions."""

    def __init__(
        self,
        filter_zeros: bool = False,
        min_prominence: float = 0.0,
        min_distance: int = 1,
        min_width: int = 1,
        top_k: int | None = None,
        max_cluster_peaks: int = 2,
        soft_top_k_margin: float = 0.90,
        drop_localization: str = "center",
        classify_return_to_baseline: bool = False,
        baseline_percentile: float = 0.20,
        baseline_abs_tolerance: float = 1.0,
        baseline_rel_tolerance: float = 0.25,
        reversion_margin: float = 1.0,
        persistence_window: int = 2,
    ) -> None:
        super().__init__(filter_zeros=filter_zeros)
        self.min_prominence = min_prominence
        self.min_distance = max(1, min_distance)
        self.min_width = max(1, min_width)
        self.top_k = top_k
        self.max_cluster_peaks = max(1, max_cluster_peaks)
        self.soft_top_k_margin = min(max(float(soft_top_k_margin), 0.0), 1.0)
        self.drop_localization = drop_localization
        self.classify_return_to_baseline = classify_return_to_baseline
        self.baseline_percentile = min(max(float(baseline_percentile), 0.0), 1.0)
        self.baseline_abs_tolerance = max(0.0, float(baseline_abs_tolerance))
        self.baseline_rel_tolerance = max(0.0, float(baseline_rel_tolerance))
        self.reversion_margin = max(0.0, float(reversion_margin))
        self.persistence_window = max(1, int(persistence_window))
        self.duplicate_radius = max(1, min(3, self.min_distance // 5 if self.min_distance > 1 else 1))

    def _detect(self, filtered_signal: np.ndarray, indices: np.ndarray) -> list[DetectionResult]:
        candidates = self._collect_candidates(filtered_signal, indices)
        results = self._select_results(candidates)
        results.sort(key=lambda result: int(result.spike_minute))
        return results

    def _collect_candidates(
        self,
        filtered_signal: np.ndarray,
        indices: np.ndarray,
    ) -> list[dict[str, float | int | str]]:
        if len(filtered_signal) < 2:
            return []
        baseline_level = float(np.percentile(filtered_signal, self.baseline_percentile * 100.0))

        candidates: list[dict[str, float | int | str]] = []
        edge_idx = 0
        while edge_idx < len(filtered_signal) - 1:
            edge_drop = float(filtered_signal[edge_idx] - filtered_signal[edge_idx + 1])
            if edge_drop <= self.min_prominence:
                edge_idx += 1
                continue
            if edge_idx > 0 and float(filtered_signal[edge_idx - 1] - filtered_signal[edge_idx]) > 0.0:
                edge_idx += 1
                continue

            run_start = edge_idx
            run_end = edge_idx
            total_drop = edge_drop
            while run_end < len(filtered_signal) - 2:
                next_drop = float(filtered_signal[run_end + 1] - filtered_signal[run_end + 2])
                if next_drop <= 0.0:
                    break
                run_end += 1
                total_drop += next_drop

            pre_start = max(0, run_start + 1 - self.persistence_window)
            pre_window = filtered_signal[pre_start:run_start + 1]
            post_start = run_end + 1
            post_stop = min(len(filtered_signal), post_start + self.persistence_window)
            post_window = filtered_signal[post_start:post_stop]
            if len(pre_window) == 0 or len(post_window) == 0:
                edge_idx = run_end + 1
                continue

            pre_level = float(np.max(pre_window))
            post_level = float(np.median(post_window))
            valley_value = float(np.min(post_window))
            minute = self._localize_drop(
                run_end,
                {"left_edge_idx": run_start},
                filtered_signal,
                indices,
            )

            drop_depth = max(0.0, pre_level - valley_value)
            sustained_drop = max(0.0, pre_level - post_level)
            relative_drop = sustained_drop / max(abs(pre_level), 1.0)
            width = float(len(post_window))

            event_type = "drop"
            if self.classify_return_to_baseline and self._is_return_to_baseline(
                baseline_level=baseline_level,
                pre_level=pre_level,
                post_level=post_level,
                sustained_drop=sustained_drop,
            ):
                event_type = "return_to_baseline"

            candidates.append(
                {
                    "event_type": event_type,
                    "minute": minute,
                    "prominence": total_drop,
                    "width": width,
                    "drop_depth": drop_depth,
                    "sustained_drop": sustained_drop,
                    "relative_drop": relative_drop,
                }
            )
            edge_idx = run_end + 1
        return candidates

    def _select_results(self, candidates: list[dict[str, float | int | str]]) -> list[DetectionResult]:
        if not candidates:
            return []

        minutes = np.asarray([float(candidate["minute"]) for candidate in candidates], dtype=float)
        prominences = np.asarray([float(candidate["prominence"]) for candidate in candidates], dtype=float)
        widths = np.asarray([float(candidate["width"]) for candidate in candidates], dtype=float)
        drop_depths = np.asarray([float(candidate["drop_depth"]) for candidate in candidates], dtype=float)
        sustained_drops = np.asarray([float(candidate["sustained_drop"]) for candidate in candidates], dtype=float)
        relative_drops = np.asarray([float(candidate["relative_drop"]) for candidate in candidates], dtype=float)

        drop_cutoff = self._local_drop_threshold(np.maximum(drop_depths, sustained_drops))
        scores = sustained_drops + 0.25 * widths + 0.10 * prominences

        duplicate_clusters = self._cluster_ids(minutes, radius=self.duplicate_radius)
        accepted: set[int] = set()
        cluster_members: dict[int, list[int]] = {}
        for idx, cluster_id in enumerate(duplicate_clusters):
            cluster_members.setdefault(int(cluster_id), []).append(idx)

        for cluster_id in sorted(cluster_members):
            members = cluster_members[cluster_id]
            qualifying = [
                idx for idx in members
                if sustained_drops[idx] >= drop_cutoff - 1e-12 or relative_drops[idx] >= 0.15 - 1e-12
            ]
            if not qualifying:
                continue

            qualifying.sort(key=lambda idx: (-scores[idx], -sustained_drops[idx], -widths[idx], minutes[idx]))
            keep_n = min(self.max_cluster_peaks, len(qualifying))
            boundary_score = float(scores[qualifying[keep_n - 1]])
            accepted.update(idx for idx in qualifying if scores[idx] >= boundary_score - 1e-12)

        if self.top_k is not None and len(accepted) > self.top_k:
            accepted_list = sorted(accepted, key=lambda idx: (-scores[idx], -sustained_drops[idx], -widths[idx], minutes[idx]))
            boundary_score = float(scores[accepted_list[self.top_k - 1]])
            override_cutoff = boundary_score * self.soft_top_k_margin
            accepted = {idx for idx in accepted_list if scores[idx] >= override_cutoff - 1e-12}

        return [
            DetectionResult(
                event_type=str(candidates[idx]["event_type"]),
                spike_minute=int(candidates[idx]["minute"]),
                score=float(scores[idx]),
            )
            for idx in accepted
        ]

    def _is_return_to_baseline(
        self,
        baseline_level: float,
        pre_level: float,
        post_level: float,
        sustained_drop: float,
    ) -> bool:
        baseline_tolerance = max(
            self.baseline_abs_tolerance,
            abs(baseline_level) * self.baseline_rel_tolerance,
        )
        post_near_baseline = abs(post_level - baseline_level) <= baseline_tolerance
        was_meaningfully_elevated = pre_level >= baseline_level + max(self.reversion_margin, baseline_tolerance)
        return post_near_baseline and was_meaningfully_elevated and sustained_drop >= self.reversion_margin

    @staticmethod
    def _cluster_ids(minutes: np.ndarray, radius: int) -> np.ndarray:
        if len(minutes) == 0:
            return np.zeros(0, dtype=int)
        order = np.argsort(minutes)
        cluster_ids = np.zeros(len(minutes), dtype=int)
        cluster = 0
        prev_minute = float(minutes[order[0]])
        cluster_ids[order[0]] = cluster
        for idx in order[1:]:
            minute = float(minutes[idx])
            if minute - prev_minute > radius:
                cluster += 1
            cluster_ids[idx] = cluster
            prev_minute = minute
        return cluster_ids

    @staticmethod
    def _local_drop_threshold(drop_strengths: np.ndarray) -> float:
        if len(drop_strengths) == 0:
            return 0.0
        q25, q50, q75 = np.percentile(drop_strengths, [25, 50, 75])
        iqr = float(q75 - q25)
        return max(0.0, max(float(q25), float(q50 - 0.25 * iqr)))

    def _localize_drop(
        self,
        peak_idx: int,
        properties: dict[str, int],
        filtered_signal: np.ndarray,
        indices: np.ndarray,
    ) -> int:
        if self.drop_localization == "left_edge":
            return int(indices[properties["left_edge_idx"]])

        if self.drop_localization == "left_ips":
            try:
                left_ips_arr = peak_widths(-filtered_signal, [peak_idx], rel_height=1.0)[2]
                left_idx = int(np.clip(round(float(left_ips_arr[0])), 0, len(indices) - 1))
                return int(indices[left_idx])
            except Exception:
                return int(indices[min(peak_idx + 1, len(indices) - 1)])

        return int(indices[min(peak_idx + 1, len(indices) - 1)])
