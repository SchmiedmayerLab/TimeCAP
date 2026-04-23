import unittest
import sys
import types

import numpy as np


datasets_stub = types.ModuleType("datasets")
datasets_stub.load_from_disk = lambda *args, **kwargs: None
sys.modules.setdefault("datasets", datasets_stub)

pyarrow_stub = types.ModuleType("pyarrow")
pyarrow_compute_stub = types.ModuleType("pyarrow.compute")
pyarrow_compute_stub.less_equal = lambda *args, **kwargs: None
pyarrow_stub.compute = pyarrow_compute_stub
sys.modules.setdefault("pyarrow", pyarrow_stub)
sys.modules.setdefault("pyarrow.compute", pyarrow_compute_stub)

torch_stub = types.ModuleType("torch")
torch_utils_stub = types.ModuleType("torch.utils")
torch_utils_data_stub = types.ModuleType("torch.utils.data")


class _Dataset:
    pass


torch_utils_data_stub.Dataset = _Dataset
torch_utils_stub.data = torch_utils_data_stub
torch_stub.utils = torch_utils_stub
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.utils", torch_utils_stub)
sys.modules.setdefault("torch.utils.data", torch_utils_data_stub)

scipy_stub = types.ModuleType("scipy")
scipy_signal_stub = types.ModuleType("scipy.signal")
scipy_signal_stub.find_peaks = lambda *args, **kwargs: (np.array([], dtype=int), {})
scipy_stub.signal = scipy_signal_stub
sys.modules.setdefault("scipy", scipy_stub)
sys.modules.setdefault("scipy.signal", scipy_signal_stub)

from extractors.cross_channel import (
    build_default_cross_channel_extractor,
    build_default_cross_channel_synthesizers,
)
from mhc.constants import MHC_CHANNEL_CONFIG
from mhc_weekly.constants import WEEKLY_CHANNEL_CONFIG
from timef.schema import Recording


def _make_recording(config, row_id: str) -> Recording:
    data = np.full((len(config.names), 168), np.nan, dtype=np.float32)

    def set_window(channel_name: str, start: int, end: int, value: float) -> None:
        idx = config.names.index(channel_name)
        data[idx, start:end] = value

    set_window("sleep:inbed", 10, 13, 1.0)
    set_window("sleep:asleep", 11, 12, 1.0)

    set_window("workout:HKWorkoutActivityTypeRunning", 30, 33, 1.0)
    set_window("hk_watch:HKQuantityTypeIdentifierHeartRate", 30, 33, 120.0)
    set_window("hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning", 30, 33, 800.0)
    set_window("hk_watch:HKQuantityTypeIdentifierStepCount", 30, 33, 1200.0)
    set_window("hk_watch:HKQuantityTypeIdentifierHeartRate", 0, 168, 90.0)

    set_window("workout:HKWorkoutActivityTypeHighIntensityIntervalTraining", 50, 52, 1.0)
    set_window("hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning", 50, 52, 2.0)
    set_window("hk_watch:HKQuantityTypeIdentifierStepCount", 50, 52, 10.0)
    set_window("hk_watch:HKQuantityTypeIdentifierHeartRate", 50, 52, 130.0)

    has_any_data = np.array([bool(np.any(np.isfinite(channel))) for channel in data], dtype=bool)
    channel_variance = np.zeros(len(config.names), dtype=np.float32)
    for idx, channel in enumerate(data):
        valid = channel[np.isfinite(channel)]
        if valid.size > 0:
            channel_variance[idx] = float(np.var(valid))

    return Recording(
        row_id=row_id,
        user_id="user-1",
        date="2026-01-05",
        values=data,
        channel_names=tuple(config.names),
        display_names=tuple(config.display_name(name) for name in config.names),
        units=tuple(config.meta.get(name, ("", None, 0))[1] for name in config.names),
        has_any_data=has_any_data,
        channel_variance=channel_variance,
    )


class WeeklyCrossChannelTests(unittest.TestCase):
    def test_shared_factory_uses_variant_specific_min_duration(self) -> None:
        daily = build_default_cross_channel_synthesizers(MHC_CHANNEL_CONFIG)
        weekly = build_default_cross_channel_synthesizers(WEEKLY_CHANNEL_CONFIG)

        self.assertTrue(all(synth.min_duration == 5 for synth in daily))
        self.assertTrue(all(synth.min_duration == 1 for synth in weekly))

    def test_weekly_cross_channel_annotations_use_hourly_wording(self) -> None:
        row = _make_recording(WEEKLY_CHANNEL_CONFIG, "weekly-row")
        extractor = build_default_cross_channel_extractor(WEEKLY_CHANNEL_CONFIG)

        annotations = extractor.extract(row)

        labels = {annotation.label for annotation in annotations}
        self.assertIn("in_bed_not_sleeping", labels)
        self.assertIn("cardio_running", labels)
        self.assertIn("stationary_hiit", labels)

        text = " ".join(annotation.text.lower() for annotation in annotations)
        self.assertIn("hour", text)
        self.assertIn("steps/hr", text)
        self.assertIn("m/hr", text)
        self.assertIn("week's mean", text)
        self.assertNotIn("steps/min", text)
        self.assertNotIn("m/min", text)
        self.assertNotIn("day's mean", text)
        self.assertNotIn("minute", text)


if __name__ == "__main__":
    unittest.main()
