import cv2

class EmergencyVehicleDetector:
    """
    Detect emergency vehicles from YOLO detections.
    Current implementation uses vehicle labels and can be extended
    with a custom ambulance/fire-truck model later.
    """

    def __init__(self):
        self.emergency_labels = {
            "ambulance",
            "fire truck",
            "firetruck",
            "police car",
            "police"
        }

    def detect(self, detections, frame_width):
        emergency_detected = False
        emergency_lane = None
        emergency_type = None

        for det in detections:
            label = det.get("label", "").lower()

            if label in self.emergency_labels:
                emergency_detected = True
                emergency_type = label

                x1, y1, x2, y2 = det["bbox"]
                center_x = (x1 + x2) // 2

                lane = min(center_x * 4 // frame_width, 3) + 1
                emergency_lane = lane

                break

        return {
            "detected": emergency_detected,
            "lane": emergency_lane,
            "type": emergency_type
        }

    def draw(self, frame, emergency_info):
        if emergency_info["detected"]:
            text = (
                f"EMERGENCY: {emergency_info['type']} | "
                f"Lane {emergency_info['lane']}"
            )

            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

        return frame