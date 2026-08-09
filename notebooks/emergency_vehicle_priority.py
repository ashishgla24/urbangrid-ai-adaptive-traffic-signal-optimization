import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8s.pt")

# Open traffic video
cap = cv2.VideoCapture("data/raw/traffic.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# COCO vehicle class IDs
vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Ambulance keywords (YOLO may classify ambulances as trucks, buses, or cars)
ambulance_keywords = ["ambulance", "emergency"]

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.resize(frame, (960, 540))

    h, w, _ = frame.shape

    results = model(frame, conf=0.15, imgsz=960)

    emergency_detected = False

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])

            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw normal vehicle boxes
            if cls in vehicle_ids:
                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (0, 255, 0),
                              2)

                cv2.putText(frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2)

            # Emergency detection (future-ready)
            if any(word in label.lower() for word in ambulance_keywords):
                emergency_detected = True

    # ----------------------------------------------------
    # DEMO SIMULATION:
    # If more than 3 buses are visible, simulate ambulance arrival
    # ----------------------------------------------------
    bus_count = 0

    for result in results:
        for box in result.boxes:
            if model.names[int(box.cls[0])] == "bus":
                bus_count += 1

    if bus_count >= 3:
        emergency_detected = True

    # ----------------------------------------------------
    # Signal Control
    # ----------------------------------------------------
    if emergency_detected:

        signal_color = (0, 255, 0)  # Green

        signal_text = "GREEN CORRIDOR ACTIVATED"

        status_text = "EMERGENCY VEHICLE PRIORITY"

    else:

        signal_color = (0, 0, 255)  # Red

        signal_text = "NORMAL SIGNAL OPERATION"

        status_text = "STANDARD TRAFFIC CONTROL"

    # Draw traffic signal
    cv2.circle(frame, (900, 60), 25, signal_color, -1)

    cv2.putText(frame,
                signal_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                signal_color,
                2)

    cv2.putText(frame,
                status_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                signal_color,
                2)

    cv2.imshow("UrbanGrid AI - Emergency Vehicle Priority", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()