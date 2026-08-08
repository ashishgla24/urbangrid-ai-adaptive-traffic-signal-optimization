import cv2
from ultralytics import YOLO

# Load YOLOv8 small model
model = YOLO("yolov8s.pt")

# Open traffic video
video_path = "data/raw/traffic.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# COCO class IDs for vehicles
vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

while True:
    success, frame = cap.read()

    if not success:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (960, 540))

    # Run YOLO detection
    results = model(frame, conf=0.15, imgsz=960)

    vehicle_count = 0

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])

            # Keep only vehicle detections
            if cls in vehicle_ids:
                vehicle_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls]

                # Draw green bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Vehicle label
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

    # Display total vehicle count
    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    # Show output
    cv2.imshow("UrbanGrid AI - Vehicle Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()