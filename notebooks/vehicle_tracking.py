import cv2
import supervision as sv
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8s.pt")

# Initialize ByteTrack
tracker = sv.ByteTrack()

# Open video
cap = cv2.VideoCapture("data/raw/traffic.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# COCO vehicle class IDs
vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (960, 540))

    # YOLO detection
    result = model(frame, conf=0.20, imgsz=960)[0]

    detections = sv.Detections.from_ultralytics(result)

    # Filter only vehicles
    mask = [cls in vehicle_ids for cls in detections.class_id]
    detections = detections[mask]

    # Update tracker
    tracked = tracker.update_with_detections(detections)

    h, w, _ = frame.shape
    lane_width = w // 4
    lane_counts = [0, 0, 0, 0]

    # Draw lane boundaries
    for i in range(1, 4):
        cv2.line(frame, (i * lane_width, 0), (i * lane_width, h), (255, 255, 0), 2)

    # Iterate through tracked detections
    for xyxy, tracker_id, class_id in zip(
        tracked.xyxy,
        tracked.tracker_id,
        tracked.class_id,
    ):
        x1, y1, x2, y2 = map(int, xyxy)

        cx = (x1 + x2) // 2
        lane = min(cx // lane_width, 3)
        lane_counts[lane] += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {int(tracker_id)}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    queue_length = max(lane_counts)
    waiting_time = queue_length * 2
    total_flow = sum(lane_counts)

    # Display lane counts
    for i in range(4):
        cv2.putText(
            frame,
            f"Lane {i+1}: {lane_counts[i]}",
            (20, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        frame,
        f"Queue Length: {queue_length}",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Estimated Wait: {waiting_time} sec",
        (20, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Traffic Flow: {total_flow} vehicles",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )

    cv2.imshow("UrbanGrid AI - Vehicle Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()