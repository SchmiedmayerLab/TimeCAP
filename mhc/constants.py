import os

from extractors import ChannelConfig
from aggregators import NonZeroAggregator
from detectors.trend import TrendDetector
from detectors.spike import SpikeDetector

CHANNEL_NAMES = [
    "step_count_iphone", "distance_iphone", "flights_iphone",
    "step_count_watch", "distance_watch", "heart_rate", "active_energy",
    "asleep", "in_bed",
    "walking", "cycling", "running", "other", "mixed_cardio",
    "strength", "elliptical", "hiit", "functional", "yoga",
]

CHANNEL_META = {
    "step_count_iphone": ("iPhone step count",       "steps/min",  1),
    "distance_iphone":   ("iPhone distance",          "km",        3),
    "flights_iphone":    ("flights climbed (iPhone)", "flights",   1),
    "step_count_watch":  ("Apple Watch step count",   "steps/min", 1),
    "distance_watch":    ("Apple Watch distance",     "km",        3),
    "heart_rate":        ("heart rate",               "bpm",       1),
    "active_energy":     ("active energy",            "kcal",      1),
}

CONTINUOUS_CHANNELS = frozenset(CHANNEL_META.keys())

MHC_CHANNEL_CONFIG = ChannelConfig(
    names=CHANNEL_NAMES,
    meta=CHANNEL_META,
    continuous=CONTINUOUS_CHANNELS,
    aggregators={"heart_rate": NonZeroAggregator()},
    detectors={
        "step_count_iphone": [TrendDetector(), SpikeDetector()],
        "distance_iphone":   [TrendDetector(), SpikeDetector()],
        "flights_iphone":    [TrendDetector(), SpikeDetector()],
        "step_count_watch":  [TrendDetector(), SpikeDetector()],
        "distance_watch":    [TrendDetector(), SpikeDetector()],
        "heart_rate":        [TrendDetector(filter_zeros=True), SpikeDetector(filter_zeros=True)],
        "active_energy":     [TrendDetector(), SpikeDetector()],
    },
)

DATASET_DIR = os.environ.get("MHC_DATASET_DIR", "data/mhc")
