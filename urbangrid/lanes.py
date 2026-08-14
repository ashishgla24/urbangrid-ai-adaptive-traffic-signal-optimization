"""
Lane assignment via calibrated polygon zones instead of assuming
4 equal-width vertical strips of the frame. Every camera angle /
intersection needs its own calibration file; this makes that a
one-time JSON edit rather than a code change.

Run this file's `interactive_calibrate()` once per camera to click
out lane polygons and save them to config/lane_zones.json.
"""

import json
import os
from typing import List, Tuple

import cv2
import numpy as np

from config import CONFIG
from urbangrid.detection import VehicleDetection

Point = Tuple[int, int]
Polygon = List[Point]


class LaneZones:
    def __init__(self, zones: List[Polygon]):
        # zones[i] is a list of (x, y) points forming lane i's polygon
        self.zones = [np.array(z, dtype=np.int32) for z in zones]

    @classmethod
    def from_file(cls, path: str = None) -> "LaneZones":
        path = path or CONFIG.paths.lane_calibration
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No lane calibration found at {path}. "
                f"Run `python -m urbangrid.lanes --calibrate` first, "
                f"or fall back to LaneZones.default_grid(width, height)."
            )
        with open(path) as f:
            data = json.load(f)
        return cls(data["zones"])

    @classmethod
    def default_grid(cls, width: int, height: int, num_lanes: int = 4) -> "LaneZones":
        """Fallback: equal vertical strips (old behaviour), used only
        when no real calibration exists yet, so the system still runs."""
        lane_width = width // num_lanes
        zones = []
        for i in range(num_lanes):
            x0, x1 = i * lane_width, (i + 1) * lane_width
            zones.append([(x0, 0), (x1, 0), (x1, height), (x0, height)])
        return cls(zones)

    def assign(self, detections: List[VehicleDetection]) -> List[int]:
        """Return a lane index (or -1 if unmatched) per detection,
        using point-in-polygon rather than a hardcoded x-cutoff."""
        lane_ids = []
        for d in detections:
            point = (d.center_x, d.center_y)
            assigned = -1
            for idx, zone in enumerate(self.zones):
                if cv2.pointPolygonTest(zone, point, False) >= 0:
                    assigned = idx
                    break
            lane_ids.append(assigned)
        return lane_ids

    def save(self, path: str = None):
        path = path or CONFIG.paths.lane_calibration
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"zones": [z.tolist() for z in self.zones]}, f, indent=2)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        for i, zone in enumerate(self.zones):
            cv2.polylines(frame, [zone], isClosed=True, color=(255, 255, 0), thickness=2)
        return frame


def interactive_calibrate(video_path: str, num_lanes: int = 4, out_path: str = None):
    """Click `num_lanes` polygons (4 points each, press 'n' to finish
    a polygon) on the first frame of the video to build a real
    calibration file for this camera."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}")

    zones, current = [], []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))

    cv2.namedWindow("Calibrate lanes - click 4 points per lane, 'n' for next, 'q' to save")
    cv2.setMouseCallback("Calibrate lanes - click 4 points per lane, 'n' for next, 'q' to save", on_click)

    while True:
        disp = frame.copy()
        for p in current:
            cv2.circle(disp, p, 4, (0, 0, 255), -1)
        for z in zones:
            cv2.polylines(disp, [np.array(z)], True, (0, 255, 0), 2)
        cv2.imshow("Calibrate lanes - click 4 points per lane, 'n' for next, 'q' to save", disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("n") and len(current) >= 3:
            zones.append(current)
            current = []
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    LaneZones(zones).save(out_path)
    print(f"Saved {len(zones)} lane zones to {out_path or CONFIG.paths.lane_calibration}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--video", default=CONFIG.paths.raw_video)
    args = parser.parse_args()
    if args.calibrate:
        interactive_calibrate(args.video)