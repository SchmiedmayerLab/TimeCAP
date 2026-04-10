import unittest

import numpy as np

from detectors.gap import GapDetector
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
            "watch_heart_rate",
            "watch_active_energy",
            "watch_steps",
            "other",
        )
        self.config = ChannelConfig(
            names=list(self.names),
            meta={name: (name, "", 1) for name in self.names},
            continuous=frozenset(self.names),
            groups={
                "watch_device": frozenset({"watch_heart_rate", "watch_active_energy", "watch_steps"}),
            },
            detectors={
                "watch_heart_rate": [GapDetector(min_duration=2)],
                "watch_active_energy": [GapDetector(min_duration=2)],
                "watch_steps": [GapDetector(min_duration=2)],
            },
        )
        self.extractor = DeviceWearExtractor(self.config, min_nonwear_duration=2)

    def test_hr_and_active_energy_produce_annotation(self) -> None:
        values = np.array([
            [1, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ])
        annotations = self.extractor.extract(_recording(values, self.names))
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].window, (2, 3))
        self.assertEqual(annotations[0].channel_idxs, (0, 1, 2))
        self.assertEqual(annotations[0].caption_type, "watch_wear")

    def test_nonwear_requires_minimum_duration(self) -> None:
        extractor = DeviceWearExtractor(self.config, min_nonwear_duration=3)
        values = np.array([
            [1, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ])
        annotations = extractor.extract(_recording(values, self.names))
        self.assertEqual(annotations, [])

    def test_watch_requires_heart_rate_gap(self) -> None:
        values = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ])
        annotations = self.extractor.extract(_recording(values, self.names))
        self.assertEqual(annotations, [])

    def test_watch_requires_active_energy_gap(self) -> None:
        values = np.array([
            [1, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ])
        annotations = self.extractor.extract(_recording(values, self.names))
        self.assertEqual(annotations, [])

    def test_steps_do_not_substitute_for_active_energy(self) -> None:
        values = np.array([
            [1, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ])
        annotations = self.extractor.extract(_recording(values, self.names))
        self.assertEqual(annotations, [])

    def test_channels_without_gap_detector_are_ignored(self) -> None:
        config = ChannelConfig(
            names=list(self.names),
            meta={name: (name, "", 1) for name in self.names},
            continuous=frozenset(self.names),
            groups={"watch_device": frozenset({"watch_heart_rate", "watch_active_energy", "watch_steps"})},
            detectors={
                "watch_heart_rate": [GapDetector(min_duration=2)],
                "watch_active_energy": [],
                "watch_steps": [GapDetector(min_duration=2)],
            },
        )
        extractor = DeviceWearExtractor(config, min_nonwear_duration=2)
        values = np.array([
            [1, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ])
        annotations = extractor.extract(_recording(values, self.names))
        self.assertEqual(annotations, [])


if __name__ == "__main__":
    unittest.main()
