import cv2
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import os

# ----------------------------------------------------
# Load YOLO Model
# ----------------------------------------------------
model = YOLO("yolov8s.pt")

# ----------------------------------------------------
# Video Path
# ----------------------------------------------------
video_path = "data/raw/traffic.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# ----------------------------------------------------
# Video Properties
# ----------------------------------------------------
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

# ----------------------------------------------------
# Ensure output directory exists
# ----------------------------------------------------
os.makedirs("data/processed", exist_ok=True)

# ----------------------------------------------------
# Traffic Log Storage
# ----------------------------------------------------
traffic_log = []

# Vehicle class IDs in COCO
vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# ----------------------------------------------------
# Main Loop
# ----------------------------------------------------
while True:

    success, frame = cap.read()

    if not success:
        break

    frame_count += 1

    frame = cv2.resize(frame, (960, 540))

    h, w, _ = frame.shape

    # ------------------------------------------------
    # Define 4 Lane Regions
    # ------------------------------------------------
    lane1_x = w // 4
    lane2_x = w // 2
    lane3_x = (3 * w) // 4

    lane_counts = [0, 0, 0, 0]

    # Draw lane lines
    cv2.line(frame, (lane1_x, 0), (lane1_x, h), (255, 255, 255), 2)
    cv2.line(frame, (lane2_x, 0), (lane2_x, h), (255, 255, 255), 2)
    cv2.line(frame, (lane3_x, 0), (lane3_x, h), (255, 255, 255), 2)

    # ------------------------------------------------
    # YOLO Detection
    # ------------------------------------------------
    results = model(frame, conf=0.15, imgsz=960)

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])

            if cls not in vehicle_ids:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = (x1 + x2) // 2

            if center_x < lane1_x:
                lane = 0
            elif center_x < lane2_x:
                lane = 1
            elif center_x < lane3_x:
                lane = 2
            else:
                lane = 3

            lane_counts[lane] += 1

            label = model.names[cls]

            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0),
                          2)

            cv2.putText(frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

    # ------------------------------------------------
    # LIVE Traffic Logging (Every Second)
    # ------------------------------------------------
        # ------------------------------------------------
    # LIVE Traffic Logging (updates every 3 frames)
    # ------------------------------------------------
    if frame_count % 3 == 0:

        total = sum(lane_counts)

        traffic_log.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "lane1": lane_counts[0],
            "lane2": lane_counts[1],
            "lane3": lane_counts[2],
            "lane4": lane_counts[3],
            "total_traffic": total
        })

        # Update CSV immediately
        df = pd.DataFrame(traffic_log)
        df.to_csv("data/processed/traffic_dataset.csv", index=False)

        # Simple prediction
        if len(traffic_log) >= 5:
            recent = [x["total_traffic"] for x in traffic_log[-5:]]
            predicted = int(sum(recent) / len(recent))
        else:
            predicted = total

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Total={total} Predicted={predicted}")

    else:

        if len(traffic_log) >= 5:
            recent = [x["total_traffic"] for x in traffic_log[-5:]]
            predicted = int(sum(recent) / len(recent))
        elif len(traffic_log) > 0:
            predicted = traffic_log[-1]["total_traffic"]
        else:
            predicted = sum(lane_counts)

    # ------------------------------------------------
    # Display Dashboard Information on Video
    # ------------------------------------------------
    cv2.putText(frame,
                f"Lane1: {lane_counts[0]}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2)

    cv2.putText(frame,
                f"Lane2: {lane_counts[1]}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    cv2.putText(frame,
                f"Lane3: {lane_counts[2]}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2)

    cv2.putText(frame,
                f"Lane4: {lane_counts[3]}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2)

    cv2.putText(frame,
                f"Current Traffic: {sum(lane_counts)}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2)

    cv2.putText(frame,
                f"Predicted Traffic: {predicted}",
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)

    cv2.putText(frame,
                "Dashboard Sync: LIVE",
                (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2)

    cv2.imshow("UrbanGrid AI - Live Traffic Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------------------------------------------
# Final Save
# ----------------------------------------------------
df = pd.DataFrame(traffic_log)
df.to_csv("data/processed/traffic_dataset.csv", index=False)

print("Traffic dataset saved to data/processed/traffic_dataset.csv")

cap.release()
cv2.destroyAllWindows()