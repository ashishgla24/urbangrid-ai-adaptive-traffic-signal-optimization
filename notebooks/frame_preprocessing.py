import cv2

# Load video
video_path = "data/raw/traffic.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    success, frame = cap.read()

    if not success:
        break

    # Resize frame
    frame = cv2.resize(frame, (960, 540))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define Region of Interest (ROI)
    roi = frame[200:500, 100:900]

    # Display all outputs
    cv2.imshow("Original", frame)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Edges", edges)
    cv2.imshow("ROI", roi)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()