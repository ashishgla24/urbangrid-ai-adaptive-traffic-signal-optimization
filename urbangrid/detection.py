"""
Single shared implementation of vehicle detection + tracking.
Every script (dashboard, notebooks, signal controller) should import
this instead of re-implementing the YOLO loop.
"""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import CONFIG


@dataclass
class VehicleDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    label: str
    track_id: Optional[int] = None

    @property
    def center_x(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def center_y(self) -> int:
        return (self.y1 + self.y2) // 2


class VehicleDetector:
    """Wraps YOLO + ByteTrack behind one clean interface."""

    def __init__(self, model_path: str = None, confidence: float = None,
                 img_size: int = None, use_tracker: bool = True):
        det_cfg = CONFIG.detection
        self.model = YOLO(model_path or det_cfg.model_path)
        self.confidence = confidence or det_cfg.confidence
        self.img_size = img_size or det_cfg.img_size
        self.vehicle_ids = set(det_cfg.vehicle_class_ids)
        self.tracker = sv.ByteTrack() if use_tracker else None

    def detect(self, frame: np.ndarray) -> List[VehicleDetection]:
        """Run detection (and tracking, if enabled) on a single frame."""
        result = self.model(frame, conf=self.confidence, imgsz=self.img_size, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        mask = np.isin(detections.class_id, list(self.vehicle_ids))
        detections = detections[mask]

        if self.tracker is not None:
            tracked_detections = self.tracker.update_with_detections(detections)
            detections = tracked_detections if len(tracked_detections) else detections

        out = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            cls_id = int(detections.class_id[i])
            track_id = (
                int(detections.tracker_id[i])
                if self.tracker is not None and detections.tracker_id is not None
                else None
            )
            out.append(VehicleDetection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                class_id=cls_id,
                label=self.model.names[cls_id],
                track_id=track_id,
            ))
        return out

    def draw(self, frame: np.ndarray, detections: List[VehicleDetection]) -> np.ndarray:
        for d in detections:
            cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
            label = f"{d.label} #{d.track_id}" if d.track_id is not None else d.label
            cv2.putText(frame, label, (d.x1, d.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame