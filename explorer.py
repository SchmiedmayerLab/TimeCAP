#
# SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2026 This source file is part of the SensorTSLM open-source project.
#
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import argparse
import textwrap
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

from annotator import Annotator
from extractors.statistical import StatisticalExtractor
from extractors.structural import StructuralExtractor
from mhc.constants import MHC_CHANNEL_CONFIG
from mhc.dataset import MHCDataset
from mhc.transformer import MHCTransformer
from timef.schema import Recording, SignalView


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive explorer for MHC rows, signals, and detector outputs.")
    parser.add_argument("--row-index", type=int, default=0, help="Initial dataset row index.")
    parser.add_argument("--signal-index", type=int, default=0, help="Initial signal index.")
    parser.add_argument("--min-wear-pct", type=float, default=0.0, help="Minimum wear percentage filter.")
    parser.add_argument("--save-path", type=str, default=None, help="Optional snapshot path. Saves current view and exits.")
    return parser.parse_args()


def _nan_regions(arr: np.ndarray, min_length: int = 30) -> list[tuple[int, int]]:
    regions = []
    in_region = False
    for i, val in enumerate(np.isnan(arr)):
        if val and not in_region:
            start = i
            in_region = True
        elif not val and in_region:
            if i - start >= min_length:
                regions.append((start, i - 1))
            in_region = False
    if in_region and len(arr) - start >= min_length:
        regions.append((start, len(arr) - 1))
    return regions


def _format_detector_event(detector_name: str, result: object) -> str:
    event_type = getattr(result, "event_type", "event")
    score = getattr(result, "score", None)
    score_suffix = "" if score in (None, 0, 0.0) else f" score={float(score):.2f}"

    if event_type == "trend":
        return (
            f"{detector_name}: {result.direction} {result.start_minute}-{result.end_minute}{score_suffix}"
        )
    if event_type in {"spike", "drop"}:
        return f"{detector_name}: {event_type} @{result.spike_minute}{score_suffix}"
    return f"{detector_name}: {event_type}{score_suffix}"


def _truncate(text: str, max_len: int = 34) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


class SensorExplorer:
    def __init__(
        self,
        dataset: MHCDataset,
        row_index: int = 0,
        signal_index: int = 0,
    ) -> None:
        self.dataset = dataset
        self.transformer = MHCTransformer()
        self.annotator = Annotator([
            StatisticalExtractor(MHC_CHANNEL_CONFIG),
            StructuralExtractor(MHC_CHANNEL_CONFIG),
        ])

        self.row_index = min(max(0, row_index), len(self.dataset) - 1)
        self.signal_index = min(max(0, signal_index), len(MHC_CHANNEL_CONFIG.names) - 1)

        self.show_trends = True
        self.show_spikes = True
        self.show_drops = True
        self.show_nonwear = True
        self.detail_mode = "events"
        self.details_scroll = 0
        self.details_page_lines = 12
        self.hit_target_names = self._available_detector_names()
        self.hit_target = self.hit_target_names[0] if self.hit_target_names else None
        self.search_status = "Use hit< / hit> to jump to the selected detector."

        self._ignore_widget_events = False

        self.fig = plt.figure(figsize=(17, 10))
        self.ax_main = self.fig.add_axes([0.04, 0.34, 0.70, 0.62])
        self.ax_overview = self.fig.add_axes([0.04, 0.16, 0.70, 0.12], sharex=self.ax_main)
        self.ax_summary = self.fig.add_axes([0.77, 0.78, 0.21, 0.18])
        self.ax_hit_target = self.fig.add_axes([0.77, 0.715, 0.21, 0.045])
        self.ax_signal_list = self.fig.add_axes([0.77, 0.43, 0.21, 0.24])
        self.ax_details = self.fig.add_axes([0.76, 0.12, 0.22, 0.22])
        for ax in (self.ax_summary, self.ax_hit_target, self.ax_signal_list, self.ax_details):
            ax.axis("off")

        self.reset_zoom_ax = self.fig.add_axes([0.04, 0.049, 0.055, 0.036])
        self.row_slider_ax = self.fig.add_axes([0.11, 0.060, 0.57, 0.024])
        self.prev_row_ax = self.fig.add_axes([0.70, 0.051, 0.035, 0.036])
        self.next_row_ax = self.fig.add_axes([0.74, 0.051, 0.035, 0.036])
        detail_tab_specs = [
            ("stats", 0.76),
            ("events", 0.815),
            ("captions", 0.870),
            ("help", 0.925),
        ]
        self.detail_tab_buttons: dict[str, Button] = {}
        for label, x0 in detail_tab_specs:
            ax = self.fig.add_axes([x0 + 0.01, 0.375, 0.047, 0.028])
            self.detail_tab_buttons[label] = Button(ax, label)
        self.detail_up_ax = self.fig.add_axes([0.935, 0.085, 0.022, 0.028])
        self.detail_down_ax = self.fig.add_axes([0.958, 0.085, 0.022, 0.028])
        self.prev_hit_ax = self.fig.add_axes([0.76, 0.698, 0.035, 0.038])
        self.next_hit_ax = self.fig.add_axes([0.945, 0.698, 0.035, 0.038])
        self.hit_target_buttons: dict[str, Button] = {}
        if self.hit_target_names:
            n_targets = len(self.hit_target_names)
            total_gap = 0.004 * max(0, n_targets - 1)
            available_width = 0.145
            width = (available_width - total_gap) / max(1, n_targets)
            for i, detector_name in enumerate(self.hit_target_names):
                x0 = 0.805 + i * (width + 0.004)
                ax = self.fig.add_axes([x0, 0.698, width, 0.038])
                self.hit_target_buttons[detector_name] = Button(ax, detector_name.replace("Detector", "").lower())
        overlay_labels = ["trend", "spike", "drop", "nonwear"]
        self.overlay_buttons: dict[str, Button] = {}
        start_x = 0.83
        button_width = 0.035
        gap = 0.006
        for i, label in enumerate(overlay_labels):
            ax = self.fig.add_axes([start_x + i * (button_width + gap), 0.051, button_width, 0.036])
            self.overlay_buttons[label] = Button(ax, label)

        self.row_slider = Slider(
            self.row_slider_ax,
            "Row",
            0,
            len(self.dataset) - 1,
            valinit=self.row_index,
            valstep=1,
            valfmt="%d",
        )
        self.reset_zoom_button = Button(self.reset_zoom_ax, "reset")
        self.prev_row_button = Button(self.prev_row_ax, "<")
        self.next_row_button = Button(self.next_row_ax, ">")
        self.prev_hit_button = Button(self.prev_hit_ax, "hit<")
        self.next_hit_button = Button(self.next_hit_ax, "hit>")
        self.detail_up_button = Button(self.detail_up_ax, "^")
        self.detail_down_button = Button(self.detail_down_ax, "v")
        self._style_widgets()
        self._sync_widgets()

        self.row_slider.on_changed(self._on_row_slider)
        self.reset_zoom_button.on_clicked(lambda _: self.render(reset_zoom=True))
        self.prev_row_button.on_clicked(lambda _: self._set_row(self.row_index - 1))
        self.next_row_button.on_clicked(lambda _: self._set_row(self.row_index + 1))
        self.prev_hit_button.on_clicked(lambda _: self._jump_to_hit(-1))
        self.next_hit_button.on_clicked(lambda _: self._jump_to_hit(1))
        self.detail_up_button.on_clicked(lambda _: self._scroll_details(-1))
        self.detail_down_button.on_clicked(lambda _: self._scroll_details(1))
        for detector_name, button in self.hit_target_buttons.items():
            button.on_clicked(lambda _, name=detector_name: self._set_hit_target(name))
        for label, button in self.detail_tab_buttons.items():
            button.on_clicked(lambda _, name=label: self._set_detail_mode(name))
        for label, button in self.overlay_buttons.items():
            button.on_clicked(lambda _, name=label: self._on_toggle(name))
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        self.render(reset_zoom=True)

    @lru_cache(maxsize=12)
    def _load_row_bundle(self, row_index: int) -> Recording:
        row = self.dataset[row_index]
        recording = self.transformer.transform_row(row)
        recording.annotations.extend(self.annotator.annotate(recording))
        return recording

    @staticmethod
    def _available_detector_names() -> list[str]:
        names = {
            detector.__class__.__name__
            for detectors in MHC_CHANNEL_CONFIG.detectors.values()
            for detector in detectors
        }
        return sorted(names)

    def _set_row(self, row_index: int) -> None:
        row_index = min(max(0, int(row_index)), len(self.dataset) - 1)
        if row_index == self.row_index:
            return
        self.row_index = row_index
        self.details_scroll = 0
        self._sync_widgets()
        self.render(reset_zoom=True)

    def _set_signal(self, signal_index: int) -> None:
        signal_index = min(max(0, int(signal_index)), len(MHC_CHANNEL_CONFIG.names) - 1)
        if signal_index == self.signal_index:
            return
        self.signal_index = signal_index
        self.details_scroll = 0
        self._sync_widgets()
        self.render(reset_zoom=True)

    def _sync_widgets(self) -> None:
        self._ignore_widget_events = True
        self.row_slider.set_val(self.row_index)
        self.row_slider_ax.set_title(f"Row {self.row_index} / {len(self.dataset) - 1}", loc="left", fontsize=10, pad=2)
        self._ignore_widget_events = False

    def _on_row_slider(self, value: float) -> None:
        if not self._ignore_widget_events:
            self._set_row(int(value))

    def _on_toggle(self, label: str) -> None:
        if label == "trend":
            self.show_trends = not self.show_trends
        elif label == "spike":
            self.show_spikes = not self.show_spikes
        elif label == "drop":
            self.show_drops = not self.show_drops
        elif label == "nonwear":
            self.show_nonwear = not self.show_nonwear
        self._update_overlay_button_styles()
        self.render(reset_zoom=False)

    def _set_hit_target(self, detector_name: str) -> None:
        if detector_name != self.hit_target:
            self.hit_target = detector_name
            self.search_status = f"Jump target set to {detector_name}."
            self._update_hit_target_button_styles()
        self.render(reset_zoom=False)

    def _set_detail_mode(self, mode: str) -> None:
        if mode != self.detail_mode:
            self.detail_mode = mode
            self.details_scroll = 0
            self._update_detail_tab_styles()
        self.render(reset_zoom=False)

    def _on_click(self, event) -> None:
        if event.inaxes is self.ax_overview and event.ydata is not None:
            self._set_signal(int(round(event.ydata)))
        elif event.inaxes is self.ax_signal_list and event.ydata is not None:
            self._set_signal(int(round(event.ydata)))

    def _on_key_press(self, event) -> None:
        if event.key == "up":
            self._set_row(self.row_index - 1)
        elif event.key == "down":
            self._set_row(self.row_index + 1)
        elif event.key == "left":
            self._set_signal(self.signal_index - 1)
        elif event.key == "right":
            self._set_signal(self.signal_index + 1)
        elif event.key == "home":
            self.render(reset_zoom=True)
        elif event.key == "pageup":
            self._scroll_details(-1)
        elif event.key == "pagedown":
            self._scroll_details(1)
        elif event.key == "n":
            self._jump_to_hit(1)
        elif event.key == "p":
            self._jump_to_hit(-1)

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.ax_details:
            return
        direction = -1 if event.button == "up" else 1
        self._scroll_details(direction)

    def _scroll_details(self, direction: int) -> None:
        self.details_scroll += direction
        self.render(reset_zoom=False)

    def _detector_events(self, signal: SignalView) -> list[tuple[str, object]]:
        events: list[tuple[float, str, object]] = []
        for detector in MHC_CHANNEL_CONFIG.detectors.get(signal.name, []):
            detector_name = detector.__class__.__name__
            for result in detector.detect(signal.data):
                events.append((float(getattr(result, "score", 0.0)), detector_name, result))

        events.sort(key=lambda item: item[0], reverse=True)
        limit = StructuralExtractor.MAX_EVENTS_PER_SIGNAL
        if limit is not None:
            events = events[:limit]

        return [(detector_name, result) for _, detector_name, result in events]

    @lru_cache(maxsize=64)
    def _row_detector_events(self, row_index: int) -> tuple[tuple[tuple[str, object], ...], ...]:
        recording = self._load_row_bundle(row_index)
        return tuple(tuple(self._detector_events(signal)) for signal in recording.iter_channels())

    def _jump_to_hit(self, step: int) -> None:
        if self.hit_target is None:
            self.search_status = "No detector target is available."
            self.render(reset_zoom=False)
            return

        n_rows = len(self.dataset)
        n_signals = len(MHC_CHANNEL_CONFIG.names)
        flat_index = self.row_index * n_signals + self.signal_index

        for offset in range(1, n_rows * n_signals):
            candidate = (flat_index + step * offset) % (n_rows * n_signals)
            row_index, signal_index = divmod(candidate, n_signals)
            row_signal_events = self._row_detector_events(row_index)[signal_index]
            if any(detector_name == self.hit_target for detector_name, _ in row_signal_events):
                self.row_index = row_index
                self.signal_index = signal_index
                self.details_scroll = 0
                self.search_status = f"Jumped to row {row_index}, signal {signal_index} with {self.hit_target}."
                self._sync_widgets()
                self.render(reset_zoom=True)
                return

        self.search_status = f"No hits found for {self.hit_target} in the scanned dataset."
        self.render(reset_zoom=False)

    @staticmethod
    def _captions_for_signal(recording: Recording, signal_idx: int) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for annotation in recording.annotations_for_signal(signal_idx):
            if annotation.text is None:
                continue
            grouped.setdefault(annotation.caption_type, []).append(annotation.text)
        return grouped

    @staticmethod
    def _overview_matrix(recording: Recording) -> np.ma.MaskedArray:
        rows = []
        for signal in recording.iter_channels():
            arr = np.asarray(signal.data, dtype=float)
            normalized = np.full_like(arr, np.nan, dtype=float)
            valid = ~np.isnan(arr)
            if valid.any():
                values = arr[valid]
                lo = float(np.nanpercentile(values, 5))
                hi = float(np.nanpercentile(values, 95))
                if hi - lo <= 1e-12:
                    normalized[valid] = 0.5
                else:
                    normalized[valid] = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
            rows.append(normalized)
        return np.ma.masked_invalid(np.vstack(rows))

    def _style_widgets(self) -> None:
        self.row_slider.label.set_visible(False)
        self.row_slider.valtext.set_visible(False)

        for button in (
            self.reset_zoom_button,
            self.prev_row_button,
            self.next_row_button,
            self.prev_hit_button,
            self.next_hit_button,
        ):
            button.label.set_fontsize(8.5)

        for button in self.hit_target_buttons.values():
            button.label.set_fontsize(8)
        for button in self.detail_tab_buttons.values():
            button.label.set_fontsize(7.5)
        self.detail_up_button.label.set_fontsize(8)
        self.detail_down_button.label.set_fontsize(8)
        for button in self.overlay_buttons.values():
            button.label.set_fontsize(7)
        self._sync_widgets()
        self._update_hit_target_button_styles()
        self._update_detail_tab_styles()
        self._update_overlay_button_styles()

    def _overlay_state(self, label: str) -> bool:
        return {
            "trend": self.show_trends,
            "spike": self.show_spikes,
            "drop": self.show_drops,
            "nonwear": self.show_nonwear,
        }[label]

    def _update_overlay_button_styles(self) -> None:
        for label, button in self.overlay_buttons.items():
            enabled = self._overlay_state(label)
            face = "#1f4f95" if enabled else "#f7f7f7"
            edge = "#f4d35e" if enabled else "#b8c0cc"
            text = "white" if enabled else "#6b7280"
            button.ax.set_facecolor(face)
            button.ax.patch.set_edgecolor(edge)
            button.ax.patch.set_linewidth(2.4 if enabled else 1.2)
            for spine in button.ax.spines.values():
                spine.set_edgecolor(edge)
                spine.set_linewidth(2.4 if enabled else 1.2)
            button.hovercolor = "#3465a4" if enabled else "#ebeff4"
            button.label.set_color(text)
            button.label.set_fontweight("bold" if enabled else "normal")

    def _update_hit_target_button_styles(self) -> None:
        for detector_name, button in self.hit_target_buttons.items():
            active = detector_name == self.hit_target
            face = "#204a87" if active else "#f4f5f7"
            edge = "#f4d35e" if active else "#c7cdd6"
            text = "white" if active else "#5f6b7a"
            button.ax.set_facecolor(face)
            button.ax.patch.set_edgecolor(edge)
            button.ax.patch.set_linewidth(2.0 if active else 1.1)
            for spine in button.ax.spines.values():
                spine.set_edgecolor(edge)
                spine.set_linewidth(2.0 if active else 1.1)
            button.hovercolor = "#3465a4" if active else "#eaedf2"
            button.label.set_color(text)
            button.label.set_fontweight("bold" if active else "normal")

    @staticmethod
    def _build_detail_lines(title: str, lines: list[str], width: int) -> list[str]:
        rendered = [title]
        if not lines:
            rendered.append("  none")
            rendered.append("")
            return rendered
        for line in lines:
            wrapped = textwrap.wrap(line, width=width) or [""]
            rendered.extend(f"  {part}" for part in wrapped)
        rendered.append("")
        return rendered

    def _update_detail_tab_styles(self) -> None:
        for label, button in self.detail_tab_buttons.items():
            active = label == self.detail_mode
            face = "#204a87" if active else "#f4f5f7"
            edge = "#f4d35e" if active else "#c7cdd6"
            text = "white" if active else "#5f6b7a"
            button.ax.set_facecolor(face)
            button.ax.patch.set_edgecolor(edge)
            button.ax.patch.set_linewidth(2.2 if active else 1.1)
            for spine in button.ax.spines.values():
                spine.set_edgecolor(edge)
                spine.set_linewidth(2.2 if active else 1.1)
            button.hovercolor = "#3465a4" if active else "#eaedf2"
            button.label.set_color(text)
            button.label.set_fontweight("bold" if active else "normal")

    def render(self, reset_zoom: bool = False) -> None:
        recording = self._load_row_bundle(self.row_index)
        n_signals = recording.values.shape[0]
        signal = recording.signal(self.signal_index)
        detector_events = self._detector_events(signal)
        captions = self._captions_for_signal(recording, self.signal_index)
        display_name = signal.display_name
        unit = signal.unit or ""
        _, _, decimals = MHC_CHANNEL_CONFIG.meta.get(signal.name, (signal.name, "", 2))

        x = np.arange(len(signal.data))
        y = np.asarray(signal.data, dtype=float)
        valid = ~np.isnan(y)

        old_xlim = self.ax_main.get_xlim()
        old_ylim = self.ax_main.get_ylim()

        self.ax_main.clear()
        self.ax_overview.clear()
        self.ax_summary.clear()
        self.ax_hit_target.clear()
        self.ax_signal_list.clear()
        self.ax_details.clear()
        for ax in (self.ax_summary, self.ax_hit_target, self.ax_signal_list, self.ax_details):
            ax.axis("off")

        self.ax_main.plot(x[valid], y[valid], color="steelblue", linewidth=1.0, label="signal")
        if self.show_nonwear:
            for start, end in _nan_regions(y):
                self.ax_main.axvspan(start, end, color="#d62728", alpha=0.08, label="nonwear")

        for detector_name, result in detector_events:
            if result.event_type == "trend" and self.show_trends:
                color = "#4daf4a" if result.direction == "increasing" else "#ff7f00"
                label = f"{detector_name} ({result.direction})"
                self.ax_main.axvspan(result.start_minute, result.end_minute, color=color, alpha=0.18, label=label)
            elif result.event_type == "spike" and self.show_spikes:
                minute = int(result.spike_minute)
                if minute < len(y) and not np.isnan(y[minute]):
                    self.ax_main.scatter(minute, y[minute], color="#2ca02c", marker="^", s=38, zorder=4, label=detector_name)
                    self.ax_main.annotate(str(minute), (minute, y[minute]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7)
            elif result.event_type == "drop" and self.show_drops:
                minute = int(result.spike_minute)
                if minute < len(y) and not np.isnan(y[minute]):
                    self.ax_main.scatter(minute, y[minute], color="#d62728", marker="v", s=38, zorder=4, label=detector_name)
                    self.ax_main.annotate(str(minute), (minute, y[minute]), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=7)

        self.ax_main.set_title(f"Row {self.row_index}  |  {display_name}")
        self.ax_main.set_ylabel(f"{display_name}\n({unit or 'value'})")
        self.ax_main.set_xlabel("Minute of day")
        self.ax_main.grid(alpha=0.2)
        self.ax_main.margins(x=0)

        handles, labels = self.ax_main.get_legend_handles_labels()
        deduped: dict[str, object] = {}
        for handle, label in zip(handles, labels):
            deduped.setdefault(label, handle)
        if deduped:
            self.ax_main.legend(deduped.values(), deduped.keys(), loc="upper right", fontsize=8)

        matrix = self._overview_matrix(recording)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="#f1f1f1")
        self.ax_overview.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, origin="upper")
        self.ax_overview.axhspan(
            self.signal_index - 0.5,
            self.signal_index + 0.5,
            facecolor="#f4d35e",
            alpha=0.20,
            edgecolor="#f4d35e",
            linewidth=0,
        )
        self.ax_overview.axhline(self.signal_index, color="white", linewidth=2)
        self.ax_overview.set_title("Channel Overview", fontsize=10, loc="left", pad=4)
        self.ax_overview.set_xlabel("Minute of day")
        self.ax_overview.set_yticks([])
        self.ax_overview.tick_params(axis="x", labelsize=8)
        self.ax_overview.text(
            1.0,
            1.02,
            "click heatmap or signal list to change channel",
            transform=self.ax_overview.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

        valid_minutes = int(np.sum(valid))
        active_channels = recording.active_channel_count()
        total_nonwear = recording.total_nonwear_minutes
        wear_pct = recording.wear_pct
        if wear_pct is None and total_nonwear is not None:
            wear_pct = (1440.0 - float(total_nonwear)) / 1440.0 * 100.0
        stats_text = "n/a"
        if valid.any():
            values = y[valid]
            stats_text = (
                f"mean={np.mean(values):.{decimals}f}\n"
                f"std={np.std(values):.{decimals}f}\n"
                f"min={np.min(values):.{decimals}f}\n"
                f"max={np.max(values):.{decimals}f}"
            )

        detector_lines = [_format_detector_event(name, result) for name, result in detector_events]
        if not detector_lines:
            detector_lines = ["No detector events on this signal."]

        caption_lines = []
        for caption_type, values in captions.items():
            for value in values[:3]:
                caption_lines.append(f"{caption_type}: {value}")
        if not caption_lines:
            caption_lines = ["No captions for this signal."]

        self.ax_summary.set_xlim(0, 1)
        self.ax_summary.set_ylim(0, 1)
        self.ax_summary.add_patch(Rectangle((0.0, 0.72), 1.0, 0.28, facecolor="#204a87", edgecolor="none"))
        self.ax_summary.text(0.03, 0.94, "Selected Signal", color="white", fontsize=9, va="top", weight="bold")
        self.ax_summary.text(0.03, 0.80, _truncate(display_name, 26), color="white", fontsize=13, va="center", weight="bold")
        self.ax_summary.text(0.03, 0.66, _truncate(signal.name, 34), color="#35506b", fontsize=8)
        summary_lines = [
            f"row {self.row_index}   signal {self.signal_index}/{n_signals - 1}",
            f"user {_truncate(str(recording.user_id), 24)}",
            f"date {recording.date}",
            f"wear {wear_pct:.1f}%   nonwear {float(total_nonwear):.0f}m" if wear_pct is not None and total_nonwear is not None else "wear n/a",
            f"active {active_channels}/{n_signals}   valid {valid_minutes}/1440",
            f"has_data {signal.has_any_data}   nonzero_or_nan {signal.minutes_nonzero_or_nan:.0f}",
        ]
        self.ax_summary.text(
            0.03,
            0.58,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=8.1,
            family="monospace",
            color="#222222",
        )
        self.ax_hit_target.set_xlim(0, 1)
        self.ax_hit_target.set_ylim(0, 1)
        self.ax_hit_target.text(
            0.0,
            1.02,
            f"Find {self.hit_target.replace('Detector', '').lower()} hits" if self.hit_target else "Find hits",
            transform=self.ax_hit_target.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#444444",
            weight="bold",
        )
        self.ax_hit_target.text(
            1.0,
            1.02,
            _truncate(self.search_status, 40),
            transform=self.ax_hit_target.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#666666",
        )

        self.ax_signal_list.set_xlim(0, 1)
        self.ax_signal_list.set_ylim(n_signals, 0)
        self.ax_signal_list.text(
            0.0,
            1.02,
            "Signals",
            transform=self.ax_signal_list.transAxes,
            fontsize=10,
            weight="bold",
            color="#333333",
            va="bottom",
        )
        for idx, listed_signal in enumerate(recording.iter_channels()):
            y0 = idx
            is_selected = idx == self.signal_index
            is_active = bool(listed_signal.has_any_data)
            face = "#204a87" if is_selected else ("#f7f7f7" if idx % 2 == 0 else "#eeeeee")
            edge = "#10253f" if is_selected else "#d0d0d0"
            text_color = "white" if is_selected else ("#222222" if is_active else "#888888")
            self.ax_signal_list.add_patch(Rectangle((0.0, y0), 1.0, 0.92, facecolor=face, edgecolor=edge, linewidth=0.8))
            self.ax_signal_list.text(
                0.03,
                y0 + 0.46,
                f"{idx:02d}",
                va="center",
                ha="left",
                fontsize=8,
                family="monospace",
                color="#f4d35e" if is_selected else "#666666",
                weight="bold",
            )
            self.ax_signal_list.text(
                0.14,
                y0 + 0.46,
                _truncate(listed_signal.display_name, 25),
                va="center",
                ha="left",
                fontsize=8.8,
                color=text_color,
                weight="bold" if is_selected else "normal",
            )
        self.ax_signal_list.text(
            0.99,
            1.01,
            "click to select",
            transform=self.ax_signal_list.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#666666",
        )

        stats_lines = stats_text.splitlines() if stats_text != "n/a" else ["n/a"]
        if self.detail_mode == "stats":
            detail_lines = self._build_detail_lines("Stats", stats_lines, width=30)
        elif self.detail_mode == "captions":
            detail_lines = self._build_detail_lines("Captions", caption_lines, width=30)
        elif self.detail_mode == "help":
            detail_lines = self._build_detail_lines(
                "Help",
                [
                    "click a signal on the right or the overview heatmap to change channel",
                    "up/down changes row",
                    "left/right changes signal",
                    "choose a detector target above the signal list",
                    "n / p jump to next or previous hit for that detector",
                    "hit< / hit> buttons do the same",
                    "mouse wheel over details scrolls",
                    "PageUp/PageDown also scroll details",
                    "overlay buttons toggle detector layers",
                ],
                width=30,
            )
        else:
            detail_lines = self._build_detail_lines("Detector Events", detector_lines, width=30)
        detail_lines = detail_lines[:-1] if detail_lines and detail_lines[-1] == "" else detail_lines
        max_scroll = max(0, len(detail_lines) - self.details_page_lines)
        self.details_scroll = min(max(self.details_scroll, 0), max_scroll)
        visible_lines = detail_lines[self.details_scroll:self.details_scroll + self.details_page_lines]
        self.ax_details.set_xlim(0, 1)
        self.ax_details.set_ylim(0, 1)
        self.ax_details.add_patch(Rectangle((0, 0), 1, 1, facecolor="#fbfbfb", edgecolor="#dddddd"))
        track_x = 0.975
        track_y = 0.06
        track_h = 0.80
        self.ax_details.add_patch(Rectangle((track_x, track_y), 0.012, track_h, facecolor="#ececec", edgecolor="#d0d0d0"))
        if detail_lines:
            visible_frac = min(1.0, self.details_page_lines / len(detail_lines))
            thumb_h = max(0.10, track_h * visible_frac)
            available_h = track_h - thumb_h
            scroll_frac = 0.0 if max_scroll == 0 else self.details_scroll / max_scroll
            thumb_y = track_y + available_h * (1.0 - scroll_frac)
            self.ax_details.add_patch(Rectangle((track_x, thumb_y), 0.012, thumb_h, facecolor="#204a87", edgecolor="none"))
        self.ax_details.text(
            0.03,
            0.90,
            "\n".join(visible_lines),
            va="top",
            ha="left",
            fontsize=7.8,
            family="monospace",
            color="#222222",
        )

        if reset_zoom:
            self.ax_main.set_xlim(0, len(y) - 1)
        else:
            self.ax_main.set_xlim(old_xlim)
            self.ax_main.set_ylim(old_ylim)

        self.fig.canvas.draw_idle()

    def save(self, path: str) -> None:
        self.fig.savefig(path, dpi=150)


def main() -> None:
    args = _parse_args()
    dataset = MHCDataset(min_wear_pct=args.min_wear_pct)
    explorer = SensorExplorer(
        dataset=dataset,
        row_index=args.row_index,
        signal_index=args.signal_index,
    )
    if args.save_path:
        explorer.save(args.save_path)
        print(f"Saved {args.save_path}")
        plt.close(explorer.fig)
        return
    plt.show()


if __name__ == "__main__":
    main()
