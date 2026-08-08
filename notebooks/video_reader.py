import cv2
import time

# Path to the traffic video
video_path = "data/raw/traffic.mp4"

# Open the video
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Variable for FPS calculation
prev_time = 0

while True:
    # Read one frame
    success, frame = cap.read()

    # Stop when video ends
    if not success:
        print("End of video reached.")
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Display resolution
    resolution_text = f"Resolution: {width} x {height}"
    cv2.putText(frame, resolution_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display FPS
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("UrbanGrid AI - Traffic Video", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Video stopped by user.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()