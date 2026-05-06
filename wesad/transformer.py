#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from datetime import datetime, timezone
import pickle
import zipfile
from pathlib import Path

import numpy as np

from extractors import ChannelConfig
from timef.schema import Recording
from transformer import Transformer
from wesad.constants import WESAD_CHANNEL_CONFIG, WESAD_CHANNEL_META, WESAD_CHANNEL_NAMES

WESAD_PRIMARY_LABEL_CHANNELS = {
    1: "wesad:label:baseline",
    2: "wesad:label:stress",
    3: "wesad:label:amusement",
    4: "wesad:label:meditation",
}
E4_ACC_G_PER_UNIT = 1.0 / 64.0


def _compute_channel_variance(data: np.ndarray) -> np.ndarray:
    variances = np.zeros(data.shape[0], dtype=np.float32)
    for idx, channel in enumerate(data):
        valid = channel[np.isfinite(channel)]
        if valid.size > 0:
            variances[idx] = float(np.var(valid))
    return variances


def _read_e4_matrix(zf: zipfile.ZipFile, file_name: str) -> dict[str, object]:
    rows = [line.strip() for line in zf.read(file_name).decode("utf-8", errors="replace").splitlines() if line.strip()]
    start_row = [float(part.strip()) for part in rows[0].split(",")]
    rate_row = [float(part.strip()) for part in rows[1].split(",")]
    values = np.asarray([[float(part.strip()) for part in row.split(",")] for row in rows[2:]], dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    sample_rate_hz = float(rate_row[0])
    return {
        "unix_start": float(start_row[0]),
        "sample_rate_hz": sample_rate_hz,
        "values": values,
        "offset_seconds": np.arange(values.shape[0], dtype=np.float32) / sample_rate_hz,
    }


def _load_e4(subject_dir: Path, subject_id: str) -> dict[str, object]:
    zip_path = subject_dir / f"{subject_id}_E4_Data.zip"
    with zipfile.ZipFile(zip_path) as zf:
        e4 = {
            "ACC": _read_e4_matrix(zf, "ACC.csv"),
            "BVP": _read_e4_matrix(zf, "BVP.csv"),
            "EDA": _read_e4_matrix(zf, "EDA.csv"),
            "TEMP": _read_e4_matrix(zf, "TEMP.csv"),
            "HR": _read_e4_matrix(zf, "HR.csv"),
        }

        ibi_rows = [line.strip() for line in zf.read("IBI.csv").decode("utf-8", errors="replace").splitlines() if line.strip()]
        ibi_start = float(ibi_rows[0].split(",")[0].strip())
        ibi_offsets = []
        ibi_values = []
        for row in ibi_rows[1:]:
            parts = [part.strip() for part in row.split(",")]
            if len(parts) >= 2:
                ibi_offsets.append(float(parts[0]))
                ibi_values.append(float(parts[1]))
        e4["IBI"] = {
            "unix_start": ibi_start,
            "offset_seconds": np.asarray(ibi_offsets, dtype=np.float32),
            "values": np.asarray(ibi_values, dtype=np.float32),
        }
    return e4


def _find_subsequence_start(haystack: np.ndarray, needle: np.ndarray) -> int | None:
    if needle.ndim != 1 or haystack.ndim != 1 or len(needle) == 0 or len(needle) > len(haystack):
        return None
    candidates = np.where(haystack[: len(haystack) - len(needle) + 1] == needle[0])[0]
    for start in candidates:
        end = start + len(needle)
        if np.array_equal(haystack[start:end], needle):
            return int(start)
    return None


def _match_full_subsequence(raw: np.ndarray, target: np.ndarray, probe_len: int = 128) -> int | None:
    probe_len = min(probe_len, len(target))
    start = _find_subsequence_start(raw, target[:probe_len])
    if start is None:
        return None
    candidate = raw[start : start + len(target)]
    if candidate.shape[0] == len(target) and np.array_equal(candidate, target):
        return start
    return None


def _infer_alignment(payload: dict, e4: dict[str, object]) -> tuple[float, dict[str, int]]:
    pkl_acc = np.asarray(payload["signal"]["wrist"]["ACC"], dtype=np.float32)
    raw_acc = np.asarray(e4["ACC"]["values"], dtype=np.float32)
    acc_start = None
    probe_len = min(64, len(pkl_acc))
    for component in range(min(3, pkl_acc.shape[1])):
        start = _find_subsequence_start(raw_acc[:, component], pkl_acc[:probe_len, component])
        if start is None:
            continue
        candidate = raw_acc[start : start + len(pkl_acc)]
        if candidate.shape == pkl_acc.shape and np.array_equal(candidate, pkl_acc):
            acc_start = start
            break
    if acc_start is None:
        raise ValueError("Could not align pickle wrist ACC to raw E4 ACC.")

    offsets = {"ACC": acc_start}
    offset_seconds = acc_start / 32.0
    for source_name, fallback_rate in (("BVP", 64.0), ("EDA", 4.0), ("TEMP", 4.0)):
        raw_values = np.asarray(e4[source_name]["values"], dtype=np.float32).reshape(-1)
        pkl_values = np.asarray(payload["signal"]["wrist"][source_name], dtype=np.float32).reshape(-1)
        start = _match_full_subsequence(raw_values, pkl_values)
        offsets[source_name] = start if start is not None else int(round(offset_seconds * fallback_rate))

    pkl_unix_start = float(e4["ACC"]["unix_start"]) + offset_seconds
    return pkl_unix_start, offsets


def _bin_continuous(values: np.ndarray, offsets_seconds: np.ndarray, n_seconds: int) -> np.ndarray:
    result = np.full(n_seconds, np.nan, dtype=np.float32)
    valid = np.isfinite(values) & np.isfinite(offsets_seconds)
    if not np.any(valid):
        return result

    values = values[valid].astype(np.float64)
    bins = np.floor(offsets_seconds[valid]).astype(np.int64)
    keep = (bins >= 0) & (bins < n_seconds)
    if not np.any(keep):
        return result

    values = values[keep]
    bins = bins[keep]
    sums = np.bincount(bins, weights=values, minlength=n_seconds)
    counts = np.bincount(bins, minlength=n_seconds)
    nonzero = counts > 0
    result[nonzero] = (sums[nonzero] / counts[nonzero]).astype(np.float32)
    return result


def _rectified(values: np.ndarray) -> np.ndarray:
    return np.abs(values.astype(np.float32, copy=False))


def _vector_magnitude(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.linalg.norm(values, axis=1).astype(np.float32)


def _bin_label_channels(label_values: np.ndarray, n_seconds: int) -> dict[str, np.ndarray]:
    result = {channel_name: np.zeros(n_seconds, dtype=np.float32) for channel_name in WESAD_PRIMARY_LABEL_CHANNELS.values()}
    for second in range(n_seconds):
        start = second * 700
        end = min((second + 1) * 700, len(label_values))
        if start >= end:
            continue
        window = label_values[start:end]
        counts = np.bincount(window.astype(np.int64), minlength=8)
        label = int(np.argmax(counts))
        channel_name = WESAD_PRIMARY_LABEL_CHANNELS.get(label)
        if channel_name is not None:
            result[channel_name][second] = 1.0
    return result


def _metadata_lines(subject_dir: Path, subject_id: str) -> list[str]:
    lines: list[str] = []
    readme_path = subject_dir / f"{subject_id}_readme.txt"
    if readme_path.exists():
        lines.extend(line.strip() for line in readme_path.read_text(errors="replace").splitlines() if line.strip())
    return lines


def _validate_wesad_payload_shape(payload: dict, subject_id: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid WESAD payload for {subject_id}: expected dict.")
    if "signal" not in payload or "label" not in payload:
        raise ValueError(f"Invalid WESAD payload for {subject_id}: missing required keys 'signal' or 'label'.")

    signal = payload["signal"]
    if not isinstance(signal, dict) or "chest" not in signal or "wrist" not in signal:
        raise ValueError(f"Invalid WESAD payload for {subject_id}: missing 'signal.chest' or 'signal.wrist'.")

    chest = signal["chest"]
    wrist = signal["wrist"]
    if not isinstance(chest, dict) or not isinstance(wrist, dict):
        raise ValueError(f"Invalid WESAD payload for {subject_id}: chest/wrist signal blocks must be dicts.")

    chest_acc = np.asarray(chest.get("ACC", []), dtype=np.float32)
    wrist_acc = np.asarray(wrist.get("ACC", []), dtype=np.float32)
    labels = np.asarray(payload["label"], dtype=np.int16).reshape(-1)
    if chest_acc.ndim != 2 or chest_acc.shape[1] != 3:
        raise ValueError(f"Invalid WESAD payload for {subject_id}: chest ACC must have shape (N, 3).")
    if wrist_acc.ndim != 2 or wrist_acc.shape[1] != 3:
        raise ValueError(f"Invalid WESAD payload for {subject_id}: wrist ACC must have shape (N, 3).")
    if labels.size == 0:
        raise ValueError(f"Invalid WESAD payload for {subject_id}: label sequence is empty.")


def build_wesad_timef_row(subject_dir: str | Path, trust: bool = False) -> dict:
    subject_dir = Path(subject_dir)
    subject_id = subject_dir.name
    if not trust:
        raise ValueError(
            "Refusing to deserialize raw WESAD pickle without explicit trust. "
            "Set trust=True only for trusted local data."
        )
    with (subject_dir / f"{subject_id}.pkl").open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    _validate_wesad_payload_shape(payload, subject_id)

    e4 = _load_e4(subject_dir, subject_id)
    session_unix_start, _offsets = _infer_alignment(payload, e4)
    labels = np.asarray(payload["label"], dtype=np.int16).reshape(-1)
    n_seconds = int(np.ceil(len(labels) / 700.0))
    label_offsets_seconds = np.arange(len(labels), dtype=np.float32) / 700.0

    chest = payload["signal"]["chest"]
    wrist = payload["signal"]["wrist"]
    chest_acc = np.asarray(chest["ACC"], dtype=np.float32)
    wrist_acc = np.asarray(wrist["ACC"], dtype=np.float32) * E4_ACC_G_PER_UNIT
    chest_emg = np.asarray(chest["EMG"], dtype=np.float32).reshape(-1)
    chest_resp = np.asarray(chest["Resp"], dtype=np.float32).reshape(-1)
    wrist_bvp = np.asarray(wrist["BVP"], dtype=np.float32).reshape(-1)
    chest_resp_mag = _rectified(chest_resp)
    chest_acc_mag = _vector_magnitude(chest_acc)
    wrist_acc_mag = _vector_magnitude(wrist_acc)
    wrist_bvp_mag = _rectified(wrist_bvp)
    channel_series = {
        "wesad:chest:acc_x": _bin_continuous(chest_acc[:, 0], np.arange(len(chest_acc), dtype=np.float32) / 700.0, n_seconds),
        "wesad:chest:acc_y": _bin_continuous(chest_acc[:, 1], np.arange(len(chest_acc), dtype=np.float32) / 700.0, n_seconds),
        "wesad:chest:acc_z": _bin_continuous(chest_acc[:, 2], np.arange(len(chest_acc), dtype=np.float32) / 700.0, n_seconds),
        "wesad:chest:acc_mag": _bin_continuous(chest_acc_mag, np.arange(len(chest_acc_mag), dtype=np.float32) / 700.0, n_seconds),
        "wesad:chest:emg": _bin_continuous(chest_emg, np.arange(len(chest_emg), dtype=np.float32) / 700.0, n_seconds),
        "wesad:chest:eda": _bin_continuous(np.asarray(chest["EDA"], dtype=np.float32).reshape(-1), np.arange(len(chest["EDA"]), dtype=np.float32) / 700.0, n_seconds),
        "wesad:chest:temp": _bin_continuous(np.asarray(chest["Temp"], dtype=np.float32).reshape(-1), np.arange(len(chest["Temp"]), dtype=np.float32) / 700.0, n_seconds),
        "wesad:chest:resp": _bin_continuous(chest_resp_mag, np.arange(len(chest_resp_mag), dtype=np.float32) / 700.0, n_seconds),
        "wesad:wrist:acc_x": _bin_continuous(wrist_acc[:, 0], np.arange(len(wrist_acc), dtype=np.float32) / 32.0, n_seconds),
        "wesad:wrist:acc_y": _bin_continuous(wrist_acc[:, 1], np.arange(len(wrist_acc), dtype=np.float32) / 32.0, n_seconds),
        "wesad:wrist:acc_z": _bin_continuous(wrist_acc[:, 2], np.arange(len(wrist_acc), dtype=np.float32) / 32.0, n_seconds),
        "wesad:wrist:acc_mag": _bin_continuous(wrist_acc_mag, np.arange(len(wrist_acc_mag), dtype=np.float32) / 32.0, n_seconds),
        "wesad:wrist:bvp": _bin_continuous(wrist_bvp_mag, np.arange(len(wrist_bvp_mag), dtype=np.float32) / 64.0, n_seconds),
        "wesad:wrist:eda": _bin_continuous(np.asarray(wrist["EDA"], dtype=np.float32).reshape(-1), np.arange(len(wrist["EDA"]), dtype=np.float32) / 4.0, n_seconds),
        "wesad:wrist:temp": _bin_continuous(np.asarray(wrist["TEMP"], dtype=np.float32).reshape(-1), np.arange(len(wrist["TEMP"]), dtype=np.float32) / 4.0, n_seconds),
    }

    hr_offsets = np.asarray(e4["HR"]["offset_seconds"], dtype=np.float32)
    hr_values = np.asarray(e4["HR"]["values"], dtype=np.float32).reshape(-1)
    hr_start_offset = max(0.0, session_unix_start - float(e4["HR"]["unix_start"]))
    hr_rel_offsets = hr_offsets - hr_start_offset
    channel_series["wesad:e4:hr"] = _bin_continuous(hr_values, hr_rel_offsets, n_seconds)
    channel_series.update(_bin_label_channels(labels, n_seconds))

    values = np.vstack([channel_series[name] for name in WESAD_CHANNEL_NAMES]).astype(np.float32)
    has_any_data = np.array([bool(np.any(~np.isnan(channel))) for channel in values], dtype=bool)
    channel_units = [WESAD_CHANNEL_META[name][1] for name in WESAD_CHANNEL_NAMES]

    ibi_offsets = np.asarray(e4["IBI"]["offset_seconds"], dtype=np.float32)
    ibi_values = np.asarray(e4["IBI"]["values"], dtype=np.float32)
    ibi_start_offset = max(0.0, session_unix_start - float(e4["IBI"]["unix_start"]))
    keep_ibi = (ibi_offsets >= ibi_start_offset) & (ibi_offsets <= ibi_start_offset + n_seconds)
    ibi_aligned_offsets = ibi_offsets[keep_ibi] - ibi_start_offset

    date = datetime.fromtimestamp(session_unix_start, tz=timezone.utc).isoformat()
    return {
        "user_id": subject_id,
        "date": date,
        "session_unix_start": float(session_unix_start),
        "data": values.tolist(),
        "values": values.tolist(),
        "has_any_data": has_any_data.tolist(),
        "channel_names": list(WESAD_CHANNEL_NAMES),
        "channel_units": channel_units,
        "label_values": labels.tolist(),
        "label_offsets_seconds": label_offsets_seconds.tolist(),
        "e4_ibi_offsets_seconds": ibi_aligned_offsets.tolist(),
        "e4_ibi_values_seconds": ibi_values[keep_ibi].tolist(),
        "source_metadata_lines": _metadata_lines(subject_dir, subject_id),
    }


class WESADTransformer(Transformer):
    def __init__(self, config: ChannelConfig = WESAD_CHANNEL_CONFIG, trust_raw_pickle: bool = True):
        self.config = config
        self.trust_raw_pickle = trust_raw_pickle

    def transform_row(self, row: dict) -> Recording:
        if "values" not in row and "data" not in row and "subject_dir" in row:
            row = build_wesad_timef_row(row["subject_dir"], trust=self.trust_raw_pickle)

        user_id = row["user_id"]
        date = row["date"]
        session_unix_start = row["session_unix_start"]
        row_id = f"wesad:{user_id}:{session_unix_start:.3f}"
        data_values = row.get("data", row.get("values"))
        data = np.asarray(data_values, dtype=np.float32).copy()

        n_channels = len(self.config.names)
        if data.shape[0] != n_channels:
            stored_names = row.get("channel_names", [])
            stored_count = len(stored_names) if isinstance(stored_names, (list, tuple)) else "unknown"
            raise ValueError(
                f"data channel count mismatch for {row_id}: expected {n_channels}, got {data.shape[0]} "
                f"(stored channel list length: {stored_count}). "
                "The WESAD preprocessing pipeline is likely out of date or the input data layout has changed."
            )

        has_any_data = np.asarray(row.get("has_any_data", []), dtype=bool)
        if has_any_data.shape != (n_channels,):
            raise ValueError(f"has_any_data must have shape ({n_channels},) for {row_id}")

        channel_names = row.get("channel_names", [])
        channel_units = row.get("channel_units", [])
        if len(channel_names) != n_channels:
            raise ValueError(
                f"channel_names length mismatch for {row_id}: expected {n_channels}, got {len(channel_names)}"
            )
        if len(channel_units) != n_channels:
            raise ValueError(
                f"channel_units length mismatch for {row_id}: expected {n_channels}, got {len(channel_units)}"
            )

        channel_variance = _compute_channel_variance(data)
        display_names = tuple(self.config.display_name(ch) for ch in self.config.names)
        units = tuple(
            channel_units[i]
            if i < len(channel_units)
            else self.config.meta.get(ch, ("", None, 0))[1]
            for i, ch in enumerate(self.config.names)
        )

        return Recording(
            row_id=row_id,
            user_id=user_id,
            date=date,
            values=np.ascontiguousarray(data),
            channel_names=tuple(self.config.names),
            display_names=display_names,
            units=units,
            has_any_data=has_any_data,
            channel_variance=channel_variance,
        )
