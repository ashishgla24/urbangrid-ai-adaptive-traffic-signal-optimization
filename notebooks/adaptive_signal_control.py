import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Open traffic video
cap = cv2.VideoCapture("data/raw/traffic.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# COCO vehicle class IDs
vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Signal timing parameters
BASE_GREEN = 10      # minimum green time
MAX_GREEN = 45       # maximum green time

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.resize(frame, (960, 540))

    h, w, _ = frame.shape
    lane_width = w // 4

    lane_counts = [0, 0, 0, 0]

    # Draw lane boundaries
    for i in range(1, 4):
        cv2.line(frame,
                 (i * lane_width, 0),
                 (i * lane_width, h),
                 (255, 255, 0),
                 2)

    # Run YOLO detection
    results = model(frame, conf=0.15, imgsz=960)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])

            if cls in vehicle_ids:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = (x1 + x2) // 2
                lane = min(cx // lane_width, 3)

                lane_counts[lane] += 1

                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (0, 255, 0),
                              2)

    # Determine most congested lane
    max_lane = lane_counts.index(max(lane_counts))

    # Calculate adaptive green time
    green_time = min(BASE_GREEN + lane_counts[max_lane] * 2,
                     MAX_GREEN)

    # Display lane counts
    for i in range(4):
        cv2.putText(frame,
                    f"Lane {i+1}: {lane_counts[i]}",
                    (20, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2)

    # Display active green signal
    cv2.putText(frame,
                f"GREEN SIGNAL: Lane {max_lane + 1}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2)

    # Display calculated green duration
    cv2.putText(frame,
                f"Green Time: {green_time} sec",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2)

    cv2.imshow("UrbanGrid AI - Adaptive Signal Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()