#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from detectors import DetectionResult, StructuralDetector


class SpikeDetector(StructuralDetector):
    """Detects upward point anomalies using a local-baseline heuristic."""

    def __init__(
        self,
        filter_zeros: bool = False,
        min_prominence: float = 0.0,
        min_distance: int = 1,
        min_width: int = 1,
        top_k: int | None = None,
        max_cluster_peaks: int = 2,
        soft_top_k_margin: float = 0.90,
    ) -> None:
        super().__init__(filter_zeros=filter_zeros)
        self.min_prominence = min_prominence
        self.min_distance = max(1, min_distance)
        self.min_width = max(1, min_width)
        self.top_k = top_k
        self.max_cluster_peaks = max(1, max_cluster_peaks)
        self.soft_top_k_margin = min(max(float(soft_top_k_margin), 0.0), 1.0)
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
    ) -> list[dict[str, float | int]]:
        peaks, properties = find_peaks(
            filtered_signal,
            prominence=self.min_prominence,
            width=self.min_width,
        )
        if len(peaks) == 0:
            return []

        prominences = properties["prominences"].astype(float)
        left_ips = properties.get("left_ips", peaks).astype(float)
        right_ips = properties.get("right_ips", peaks).astype(float)
        left_bases = properties.get("left_bases", peaks).astype(int)
        right_bases = properties.get("right_bases", peaks).astype(int)
        widths = properties.get("widths", np.ones_like(peaks, dtype=float)).astype(float)

        candidates: list[dict[str, float | int]] = []
        for idx, peak in enumerate(peaks):
            peak_idx = int(peak)
            minute = self._localize_spike(
                peak_idx,
                {
                    "left_idx": int(np.floor(left_ips[idx])),
                    "right_idx": int(np.ceil(right_ips[idx])),
                },
                filtered_signal,
                indices,
            )
            left_base_value = float(filtered_signal[int(np.clip(left_bases[idx], 0, len(filtered_signal) - 1))])
            right_base_value = float(filtered_signal[int(np.clip(right_bases[idx], 0, len(filtered_signal) - 1))])
            baseline_value = max(left_base_value, right_base_value)
            local_jump = max(0.0, float(filtered_signal[peak_idx]) - baseline_value)
            relative_jump = local_jump / max(abs(baseline_value), 1.0)
            candidates.append(
                {
                    "minute": minute,
                    "prominence": float(prominences[idx]),
                    "width": float(widths[idx]),
                    "local_jump": local_jump,
                    "relative_jump": relative_jump,
                }
            )
        return candidates

    def _select_results(self, candidates: list[dict[str, float | int]]) -> list[DetectionResult]:
        if not candidates:
            return []

        minutes = np.asarray([float(candidate["minute"]) for candidate in candidates], dtype=float)
        prominences = np.asarray([float(candidate["prominence"]) for candidate in candidates], dtype=float)
        widths = np.asarray([float(candidate["width"]) for candidate in candidates], dtype=float)
        local_jumps = np.asarray([float(candidate["local_jump"]) for candidate in candidates], dtype=float)
        relative_jumps = np.asarray([float(candidate["relative_jump"]) for candidate in candidates], dtype=float)

        jump_cutoff = self._local_jump_threshold(local_jumps)
        scores = local_jumps + 0.25 * widths + 0.10 * prominences

        duplicate_clusters = self._cluster_ids(minutes, radius=self.duplicate_radius)
        accepted: set[int] = set()
        cluster_members: dict[int, list[int]] = {}
        for idx, cluster_id in enumerate(duplicate_clusters):
            cluster_members.setdefault(int(cluster_id), []).append(idx)

        for cluster_id in sorted(cluster_members):
            members = cluster_members[cluster_id]
            qualifying = [
                idx for idx in members
                if local_jumps[idx] >= jump_cutoff - 1e-12 or relative_jumps[idx] >= 1.0 - 1e-12
            ]
            if not qualifying:
                continue

            qualifying.sort(key=lambda idx: (-scores[idx], -local_jumps[idx], -widths[idx], minutes[idx]))
            keep_n = min(self.max_cluster_peaks, len(qualifying))
            boundary_score = float(scores[qualifying[keep_n - 1]])
            accepted.update(idx for idx in qualifying if scores[idx] >= boundary_score - 1e-12)

        if self.top_k is not None and len(accepted) > self.top_k:
            accepted_list = sorted(accepted, key=lambda idx: (-scores[idx], -local_jumps[idx], -widths[idx], minutes[idx]))
            boundary_score = float(scores[accepted_list[self.top_k - 1]])
            override_cutoff = boundary_score * self.soft_top_k_margin
            accepted = {idx for idx in accepted_list if scores[idx] >= override_cutoff - 1e-12}

        return [
            DetectionResult(
                event_type="spike",
                spike_minute=int(candidates[idx]["minute"]),
                score=float(scores[idx]),
            )
            for idx in accepted
        ]

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
    def _local_jump_threshold(local_jumps: np.ndarray) -> float:
        if len(local_jumps) == 0:
            return 0.0
        q25, q50, q75 = np.percentile(local_jumps, [25, 50, 75])
        iqr = float(q75 - q25)
        return max(0.0, max(float(q25), float(q50 - 0.25 * iqr)))

    @staticmethod
    def _localize_spike(
        peak_idx: int,
        properties: dict[str, int],
        filtered_signal: np.ndarray,
        indices: np.ndarray,
    ) -> int:
        left_idx = int(np.clip(properties.get("left_idx", peak_idx), 0, len(indices) - 1))
        right_idx = int(np.clip(properties.get("right_idx", peak_idx), left_idx, len(indices) - 1))
        window = filtered_signal[left_idx:right_idx + 1]
        if len(window) == 0:
            return int(indices[peak_idx])

        max_height = float(np.max(window))
        candidate_offsets = np.flatnonzero(window == max_height)
        if len(candidate_offsets) == 0:
            return int(indices[peak_idx])

        candidate_indices = left_idx + candidate_offsets
        best_idx = min(candidate_indices, key=lambda idx: abs(idx - peak_idx))
        return int(indices[int(best_idx)])
