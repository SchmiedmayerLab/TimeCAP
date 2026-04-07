import unittest

import numpy as np

from detectors.drop import DropDetector
from detectors.spike import SpikeDetector


class DropDetectorTests(unittest.TestCase):
    def test_return_to_baseline_for_activity_like_signal(self) -> None:
        detector = DropDetector(
            classify_return_to_baseline=True,
            drop_localization="center",
            baseline_abs_tolerance=0.5,
            reversion_margin=0.5,
        )
        signal = np.asarray([0.0, 0.0, 0.0, 12.0, 10.0, 0.0, 0.0, 0.0], dtype=float)

        results = detector.detect(signal)

        self.assertEqual(["return_to_baseline"], [result.event_type for result in results])
        self.assertEqual([5], [result.spike_minute for result in results])

    def test_continuous_short_downshift_is_drop(self) -> None:
        detector = DropDetector(
            classify_return_to_baseline=False,
            min_distance=5,
            persistence_window=2,
        )
        signal = np.asarray([80.0, 82.0, 81.0, 80.0, 79.0, 60.0, 61.0, 79.0, 80.0], dtype=float)

        results = detector.detect(signal)

        self.assertEqual(["drop"], [result.event_type for result in results])

    def test_noise_does_not_produce_drop(self) -> None:
        detector = DropDetector(classify_return_to_baseline=False)
        signal = np.asarray([70.0, 71.0, 69.0, 70.0, 71.0, 70.0, 69.0, 70.0], dtype=float)

        results = detector.detect(signal)

        self.assertEqual([], results)

    def test_nearby_drop_candidates_are_capped(self) -> None:
        detector = DropDetector(
            classify_return_to_baseline=True,
            min_distance=15,
            max_cluster_peaks=2,
            top_k=None,
            baseline_abs_tolerance=0.5,
            reversion_margin=0.5,
        )
        signal = np.zeros(12, dtype=float)
        signal[[2, 4, 6]] = [12.0, 11.0, 10.0]

        results = detector.detect(signal)

        self.assertLessEqual(len(results), 2)


class SpikeDetectorTests(unittest.TestCase):
    def test_spike_detector_only_emits_spikes(self) -> None:
        detector = SpikeDetector(min_distance=5, top_k=None)
        signal = np.asarray([0.0, 0.0, 10.0, 0.0, 0.0, 8.0, 0.0, 0.0], dtype=float)

        results = detector.detect(signal)

        self.assertTrue(results)
        self.assertEqual({"spike"}, {result.event_type for result in results})


if __name__ == "__main__":
    unittest.main()
