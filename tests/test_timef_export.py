from __future__ import annotations

import json
import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from timef import Annotation, CaptionResult, RuntimeRow
from timef.export import TimeFExportConfig, export_caption_result
from timef_persist import TimeFReader, validate_dataset


class TimeFExportTests(unittest.TestCase):
    def test_single_row_export_validates_and_marks_manifest(self) -> None:
        result = CaptionResult(rows=[_make_row()])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = export_caption_result(
                result,
                TimeFExportConfig(
                    output_root=Path(tmpdir),
                    dataset_id="mhc_export",
                    version="v1",
                ),
            )
            validate_dataset(root)
            manifest = TimeFReader(root).read_manifest()
            self.assertTrue(manifest.validated)
            self.assertEqual(manifest.sample_count, 1)
            self.assertEqual(manifest.annotation_count, 3)

    def test_two_row_export_has_stable_counts(self) -> None:
        result = CaptionResult(rows=[_make_row("row-a"), _make_row("row-b")])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = export_caption_result(
                result,
                TimeFExportConfig(output_root=Path(tmpdir), dataset_id="mhc_export", version="v2"),
            )
            reader = TimeFReader(root)
            manifest = reader.read_manifest()
            samples = list(reader.read_samples())
            annotations = list(reader.read_annotations())

            self.assertEqual(len(samples), 2)
            self.assertEqual(len(annotations), 6)
            self.assertEqual(manifest.sample_count, 2)
            self.assertEqual(len(manifest.signals), 4)
            self.assertEqual(len(manifest.signal_spec), 2)

    def test_round_trip_reader_returns_single_channel_views(self) -> None:
        result = CaptionResult(rows=[_make_row()])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = export_caption_result(
                result,
                TimeFExportConfig(output_root=Path(tmpdir), dataset_id="mhc_export", version="v3"),
            )
            reader = TimeFReader(root)
            manifest = reader.read_manifest()
            sample = next(reader.read_samples())
            frames = reader.read_signal_frame_for_sample(sample, manifest)

            self.assertEqual(len(frames), 2)
            for signal_id, rows in frames.items():
                self.assertEqual(len(rows), 4)
                self.assertIn("time_s", rows[0])
                self.assertEqual(len(rows[0]), 2)
                expected_channel = manifest.signal_spec[manifest.signals[signal_id].spec_id].channels[0]
                self.assertIn(expected_channel, rows[0])

    def test_annotation_mapping_preserves_label_and_rationale(self) -> None:
        result = CaptionResult(rows=[_make_row()])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = export_caption_result(
                result,
                TimeFExportConfig(output_root=Path(tmpdir), dataset_id="mhc_export", version="v4"),
            )
            annotations = list(TimeFReader(root).read_annotations())
            by_spec = {annotation.spec_id: annotation for annotation in annotations}

            statistical = by_spec[0]
            structural = by_spec[1]
            semantic = by_spec[2]

            self.assertEqual(statistical.answer, "Average heart rate was stable.")
            self.assertIsNone(statistical.label)
            self.assertIsNone(statistical.rationale)

            self.assertEqual(structural.answer, "A spike occurred during the morning interval.")
            self.assertIsNone(structural.label)

            self.assertIsNone(semantic.answer)
            self.assertEqual(semantic.label, "walking")
            self.assertEqual(semantic.rationale, "The user was walking during the afternoon.")

    def test_reference_json_and_missing_value_units_are_preserved(self) -> None:
        result = CaptionResult(rows=[_make_row()])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = export_caption_result(
                result,
                TimeFExportConfig(output_root=Path(tmpdir), dataset_id="mhc_export", version="v5"),
            )
            reader = TimeFReader(root)
            manifest = reader.read_manifest()
            annotations = list(reader.read_annotations())

            self.assertTrue(all(spec.unit_value is None for spec in manifest.signal_spec))
            reference = json.loads(annotations[1].samples[0].reference or "{}")
            self.assertEqual(reference["channel_names"], ["heart_rate"])
            self.assertEqual(reference["window"], [1, 3])
            self.assertEqual(reference["kind"], "structural")

    def test_generated_version_defaults_to_timestamp(self) -> None:
        result = CaptionResult(rows=[_make_row()])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = export_caption_result(
                result,
                TimeFExportConfig(output_root=Path(tmpdir), dataset_id="mhc_export"),
            )
            self.assertRegex(root.name, r"^\d{8}T\d{6}Z$")


def _make_row(row_id: str = "row-1") -> RuntimeRow:
    values = np.array(
        [
            [1.0, 2.0, math.nan, 4.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return RuntimeRow(
        row_id=row_id,
        user_id="user-1",
        date="2026-04-03",
        values=values,
        channel_names=("heart_rate", "steps"),
        display_names=("Heart Rate", "Steps"),
        units=("bpm", None),
        has_any_data=np.array([True, True]),
        minutes_nonzero_or_nan=np.array([4.0, 2.0], dtype=np.float32),
        channel_variance=np.array([1.0, 0.5], dtype=np.float32),
        total_nonwear_minutes=0.0,
        wear_pct=100.0,
        annotations=[
            Annotation(kind="statistical", text="Average heart rate was stable.", channel_idxs=(0,)),
            Annotation(
                kind="structural",
                text="A spike occurred during the morning interval.",
                channel_idxs=(0,),
                window=(1, 3),
            ),
            Annotation(
                kind="semantic",
                text="The user was walking during the afternoon.",
                channel_idxs=(1,),
                window=(2, 4),
                label="walking",
            ),
        ],
    )


if __name__ == "__main__":
    unittest.main()
