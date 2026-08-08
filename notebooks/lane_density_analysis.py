import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Open video
cap = cv2.VideoCapture("data/raw/traffic.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Vehicle class IDs
vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.resize(frame, (960, 540))

    h, w, _ = frame.shape

    # Divide frame into 4 vertical lanes
    lane_width = w // 4

    lane_counts = [0, 0, 0, 0]

    # Draw lane boundaries
    for i in range(1, 4):
        cv2.line(frame,
                 (i * lane_width, 0),
                 (i * lane_width, h),
                 (255, 255, 0),
                 2)

    # Run detection
    results = model(frame, conf=0.15, imgsz=960)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])

            if cls in vehicle_ids:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Vehicle center point
                cx = (x1 + x2) // 2

                lane_index = min(cx // lane_width, 3)

                lane_counts[lane_index] += 1

                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (0, 255, 0),
                              2)

                cv2.putText(frame,
                            model.names[cls],
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2)

    # Display lane counts
    for i in range(4):
        cv2.putText(frame,
                    f"Lane {i+1}: {lane_counts[i]}",
                    (20, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2)

    # Find most congested lane
    max_lane = lane_counts.index(max(lane_counts)) + 1

    cv2.putText(frame,
                f"Most Congested: Lane {max_lane}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2)

    cv2.imshow("UrbanGrid AI - Lane Density Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()