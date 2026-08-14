"""
Central configuration for UrbanGrid AI.
Every script/module should import from here instead of hardcoding
thresholds, paths, or model names. Override any value with an
environment variable of the same name (see load_from_env below).
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DetectionConfig:
    model_path: str = "yolov8s.pt"          # single source of truth (was yolov8n/yolov8s mismatch)
    confidence: float = 0.20                  # was 0.15 in some files, 0.20 in others
    img_size: int = 960
    frame_width: int = 960
    frame_height: int = 540
    vehicle_class_ids: Tuple[int, ...] = (2, 3, 5, 7)  # car, motorcycle, bus, truck


@dataclass
class SignalConfig:
    min_green_sec: int = 10
    max_green_sec: int = 45
    all_red_clearance_sec: int = 3            # mandatory all-red between phase switches
    max_wait_before_forced_switch_sec: int = 90  # anti-starvation cap for any lane
    vehicles_per_extra_second: float = 2.0    # green-time scaling factor
    num_lanes: int = 4


@dataclass
class ForecastConfig:
    sequence_length: int = 12                 # was 5 - too short to capture cycles
    lstm_units: int = 32
    epochs: int = 100
    batch_size: int = 16
    validation_split: float = 0.2
    model_out_path: str = "models/lstm_traffic_model.h5"


@dataclass
class PathConfig:
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    raw_video: str = "data/raw/traffic.mp4"
    processed_dataset: str = "data/processed/traffic_dataset.csv"
    lane_calibration: str = "config/lane_zones.json"


@dataclass
class AppConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    debug: bool = False                        # was hardcoded True in app.run()


def load_from_env(cfg: AppConfig = None) -> AppConfig:
    """Allow ops to override key settings via environment variables
    without touching code, e.g. UG_DEBUG=1, UG_MODEL_PATH=yolov8m.pt"""
    cfg = cfg or AppConfig()
    cfg.debug = os.getenv("UG_DEBUG", str(cfg.debug)) == "1"
    cfg.detection.model_path = os.getenv("UG_MODEL_PATH", cfg.detection.model_path)
    cfg.detection.confidence = float(os.getenv("UG_CONFIDENCE", cfg.detection.confidence))
    return cfg


CONFIG = load_from_env()