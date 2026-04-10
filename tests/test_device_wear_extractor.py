import unittest

import numpy as np

from extractors import ChannelConfig
from extractors.device_wear import DeviceWearExtractor
from timef.schema import Recording


def _recording(values: np.ndarray, names: tuple[str, ...]) -> Recording:
    n_channels = len(names)
    return Recording(
        row_id="test-row",
        user_id="user",
        date="2026-01-01",
        values=values.astype(np.float32),
        channel_names=names,
        display_names=names,
        units=tuple("" for _ in names),
        has_any_data=np.ones(n_channels, dtype=bool),
        channel_variance=np.zeros(n_channels, dtype=np.float32),
    )


class DeviceWearExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.names = (
            "watch_steps",
            "watch_distance",
            "watch_heart_rate",
            "watch_active_energy",
            "other",
        )
        self.config = ChannelConfig(
            names=list(self.names),
            meta={name: (name, "", 1) for name in self.names},
            continuous=frozenset(self.names),
            groups={
                "watch_device": frozenset({
                    "watch_steps",
                    "watch_distance",
                    "watch_heart_rate",
                    "watch_active_energy",
                }),
            },
        )
        self.extractor = DeviceWearExtractor(self.config, min_nonwear_duration=2)

    def test_hr_and_active_energy_overlap_produces_annotation(self) -> None:
        values = np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [70, np.nan, np.nan, 72, 73, 74],
            [5, 0, 0, 8, 9, 10],
            [1, 1, 1, 1, 1, 1],
        ])
        annotations = self.extractor.extract(_recording(values, self.names))
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].window, (1, 2))
        self.assertEqual(annotations[0].channel_idxs, (0, 1, 2, 3))
        self.assertEqual(annotations[0].caption_type, "watch_wear")

    def test_nonwear_requires_minimum_duration(self) -> None:
        extractor = DeviceWearExtractor(self.config, min_nonwear_duration=3)
        values = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [70, np.nan, 72, 73],
            [5, 0, 7, 8],
            [1, 1, 1, 1],
        ])
        annotations = extractor.extract(_recording(values, self.names))
        self.assertEqual(annotations, [])

    def test_watch_requires_heart_rate_gap(self) -> None:
        values = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [70, 71, 72, 73],
            [5, 0, 0, 8],
            [1, 1, 1, 1],
        ])
        annotations = self.extractor.extract(_recording(values, self.names))
        self.assertEqual(annotations, [])

    def test_watch_requires_active_energy_gap(self) -> None:
        values = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [70, np.nan, np.nan, 73],
            [5, 6, 7, 8],
            [1, 1, 1, 1],
        ])
        annotations = self.extractor.extract(_recording(values, self.names))
        self.assertEqual(annotations, [])

    def test_steps_do_not_substitute_for_active_energy(self) -> None:
        values = np.array([
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [70, np.nan, np.nan, 73],
            [5, 6, 7, 8],
            [1, 1, 1, 1],
        ])
        annotations = self.extractor.extract(_recording(values, self.names))
        self.assertEqual(annotations, [])

    def test_missing_watch_group_returns_no_annotations(self) -> None:
        config = ChannelConfig(
            names=list(self.names),
            meta={name: (name, "", 1) for name in self.names},
            continuous=frozenset(self.names),
            groups={},
        )
        extractor = DeviceWearExtractor(config, min_nonwear_duration=2)
        values = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [70, np.nan, np.nan, 73],
            [5, 0, 0, 8],
            [1, 1, 1, 1],
        ])
        annotations = extractor.extract(_recording(values, self.names))
        self.assertEqual(annotations, [])


if __name__ == "__main__":
    unittest.main()
