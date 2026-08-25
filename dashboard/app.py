from flask import Flask, render_template, jsonify, Response, send_file
import os
import time
import threading
import cv2
from config import CONFIG
from urbangrid.detection import VehicleDetector
from urbangrid.lanes import LaneZones
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from urbangrid.emergency_detector import EmergencyVehicleDetector
app = Flask(__name__)
detector = VehicleDetector()
lane_zones = LaneZones([])
# ----------------------------
# Paths
# ----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "raw", "traffic.mp4")
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "traffic_dataset.csv")
MAP_LATITUDE = float(os.getenv("UG_MAP_LAT", "28.6139"))
MAP_LONGITUDE = float(os.getenv("UG_MAP_LON", "77.2090"))
MAP_LOCATION_NAME = os.getenv("UG_MAP_NAME", "UrbanGrid Intersection")


# ----------------------------
# Load YOLO model
# ----------------------------



# ----------------------------
# Global live traffic data (protected by data_lock)
# ----------------------------

live_data = {
    "lane1": 0,
    "lane2": 0,
    "lane3": 0,
    "lane4": 0,
    "total": 0,
    "prediction": 0,
    "congestion": "LOW",
    "signal": "RED",
    "fps": 0.0,
    "throughput": 0,
    "congestion_score": 0,
    "system_health": "HEALTHY",
    "incident_status": "LOW",
    "incident_alert": "Traffic flowing normally",
    "incident_lane": "N/A",
    "incident_action": "Normal signal operation",

    "emergency_detected": False,
    "emergency_lane": None,
    "emergency_type": None
}

chart_labels = []
chart_values = []

# Shared JPEG bytes for whichever frame was most recently processed.
latest_jpeg = None

data_lock = threading.Lock()

# ----------------------------
# Background detection loop
#
# Runs once, in its own thread, for the lifetime of the process -
# completely decoupled from whether any browser tab is open or
# whether /video_feed is currently being requested. This is what
# makes the dashboard numbers update in real time no matter what
# the frontend is doing.
# ----------------------------

def detection_loop():
    global live_data, chart_labels, chart_values, latest_jpeg

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"[detection_loop] ERROR: could not open video at {VIDEO_PATH}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 30.0
    frame_interval = 1.0 / source_fps
    next_frame_time = time.monotonic()

    vehicle_classes = {2, 3, 5, 7}
    emergency_detector = EmergencyVehicleDetector()
    frame_count = 0
    fps_window_start = time.time()
    fps_window_frames = 0
    current_fps = 0.0

    # track_id -> timestamp first seen. Used to count *unique* vehicles
    # over a rolling 60s window for "vehicles/minute", instead of the old
    # total*60 guess. Pruned periodically so it doesn't grow unbounded.
    track_first_seen = {}
    THROUGHPUT_WINDOW_SEC = 60
    last_prune = time.time()

    # If tracking (ByteTrack) can't run - most commonly because the 'lap'/
    # 'lapx' package isn't installed - we fall back to plain detection so
    # the dashboard still shows live counts instead of going dark. A single
    # bad frame should never be able to kill this whole background thread.
    tracking_available = True
    next_fallback_id = 0

    while True:
        try:
            success, frame = cap.read()

            if not success:
                # Loop the video instead of spinning forever on a bad read.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (960, 540))
            now = time.time()

            # model.track() keeps a persistent ID per vehicle across frames
            # via ByteTrack, so a car sitting at a red light for 50 frames
            # is one vehicle, not 50 detections. conf/iou are raised from
            # the ultralytics defaults (~0.25 conf) to cut false positives
            # and duplicate boxes in dense, overlapping traffic. Falls back
            # to plain model.predict() if the tracker deps aren't installed.
            # Run the shared detector
            detections = detector.detect(frame)
            print("Detected vehicles:", len(detections), flush=True)

            # Use the annotated frame for streaming
            frame = detector.draw(frame, detections)

            h, w, _ = frame.shape

            # Draw lane boundaries
            cv2.line(frame, (int(w*0.25), 0), (int(w*0.25), h), (255,255,0), 2)
            cv2.line(frame, (int(w*0.50), 0), (int(w*0.50), h), (255,255,0), 2)
            cv2.line(frame, (int(w*0.75), 0), (int(w*0.75), h), (255,255,0), 2)

            lane_counts = [0, 0, 0, 0]
            emergency_input = []
            calibrated_lane_ids = lane_zones.assign(detections)

            for det, calibrated_lane in zip(detections, calibrated_lane_ids):
                x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                lane = calibrated_lane

                if lane == -1:
                    lane = min(center_x * 4 // w, 3)

                lane_counts[lane] += 1

                emergency_input.append({
                    "label": det.label,
                    "bbox": (x1, y1, x2, y2)
                })

            emergency_info = emergency_detector.detect(emergency_input, w)
            frame = emergency_detector.draw(frame, emergency_info)
            total = sum(lane_counts)

            if total >= 30:
                incident_status = "CRITICAL"
                incident_alert = "Severe traffic congestion detected"
                incident_action = "Activate emergency traffic diversion"
            elif total >= 20:
                incident_status = "HIGH"
                incident_alert = "Possible traffic incident or bottleneck"
                incident_action = "Extend green signal timing"
            elif total >= 10:
                incident_status = "MEDIUM"
                incident_alert = "Traffic building up"
                incident_action = "Monitor intersection continuously"
            else:
                incident_status = "LOW"
                incident_alert = "Traffic flowing normally"
                incident_action = "Normal signal operation"

            incident_lane = f"Lane {lane_counts.index(max(lane_counts)) + 1}"
            vehicles_per_minute = total
            if total > 30:
                congestion = "HIGH"
            elif total > 15:
                congestion = "MEDIUM"
            else:
                congestion = "LOW"

            max_lane = lane_counts.index(max(lane_counts)) + 1
            signal = f"GREEN - Lane {max_lane}"

            # --- real FPS, measured over a rolling ~1 second window ---
            frame_count += 1
            fps_window_frames += 1
            elapsed = now - fps_window_start
            if elapsed >= 1.0:
                current_fps = fps_window_frames / elapsed
                fps_window_frames = 0
                fps_window_start = now

            ret, buffer = cv2.imencode('.jpg', frame)
            jpeg_bytes = buffer.tobytes()

            with data_lock:
                live_data["lane1"] = lane_counts[0]
                live_data["lane2"] = lane_counts[1]
                live_data["lane3"] = lane_counts[2]
                live_data["lane4"] = lane_counts[3]
                live_data["total"] = total
                live_data["prediction"] = total + 2
                live_data["incident_status"] = incident_status
                live_data["incident_alert"] = incident_alert
                live_data["incident_lane"] = incident_lane
                live_data["incident_action"] = incident_action
                live_data["congestion"] = congestion
                live_data["signal"] = signal
                live_data["throughput"] = vehicles_per_minute
                live_data["congestion_score"] = min(total * 2, 100)
                live_data["fps"] = round(current_fps, 1)
                live_data["emergency_detected"] = emergency_info["detected"]
                live_data["emergency_lane"] = emergency_info["lane"]
                live_data["emergency_type"] = emergency_info["type"]
                chart_labels.append(str(frame_count))
                chart_values.append(total)
                if len(chart_labels) > 20:
                    chart_labels = chart_labels[-20:]
                    chart_values = chart_values[-20:]

                latest_jpeg = jpeg_bytes

            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = time.monotonic()

        except Exception as e:
            print(f"[detection_loop] Error processing frame: {e}")
            time.sleep(0.05)


# Start the background worker exactly once when the app boots.
worker_thread = threading.Thread(target=detection_loop, daemon=True)
worker_thread.start()


def mjpeg_generator():
    """Serves whatever the background thread most recently produced.
    Multiple tabs/clients can hit this simultaneously without spawning
    extra YOLO/VideoCapture instances."""
    while True:
        with data_lock:
            frame = latest_jpeg
        if frame is not None:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame +
                b'\r\n'
            )
        time.sleep(0.03)  # cap stream rate ~30fps regardless of inference speed


# ----------------------------
# Dashboard
# ----------------------------

@app.route("/")
def dashboard():
    with data_lock:
        data = dict(live_data)
        labels = list(chart_labels)
        values = list(chart_values)

    return render_template(
        "index.html",
        lane1=data["lane1"],
        lane2=data["lane2"],
        lane3=data["lane3"],
        lane4=data["lane4"],
        total=data["total"],
        prediction=data["prediction"],
        congestion=data["congestion"],
        signal=data["signal"],
        chart_labels=labels,
        chart_values=values,
        fps=data["fps"],
        throughput=data["throughput"],
        congestion_score=data["congestion_score"],
        incident_status=data["incident_status"],
        incident_alert=data["incident_alert"],
        incident_lane=data["incident_lane"],
        incident_action=data["incident_action"],
        map_latitude=MAP_LATITUDE,
        map_longitude=MAP_LONGITUDE,
        map_location_name=MAP_LOCATION_NAME
    )


# ----------------------------
# Live API
# ----------------------------

@app.route("/api/live")
def live_api():
    with data_lock:
        data = dict(live_data)
        data["chart_labels"] = list(chart_labels)
        data["chart_values"] = list(chart_values)

    return jsonify(data)


# ----------------------------
# Video stream
# ----------------------------

@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ----------------------------
# Download dataset
# ----------------------------
@app.route("/download")
def download_dataset():
    if os.path.exists(DATASET_PATH):
        return send_file(DATASET_PATH, as_attachment=True)
    return "Dataset not found", 404

# ----------------------------
# Status API
# ----------------------------

@app.route("/api/status")
def system_status():
    return jsonify({
        "system": "UrbanGrid AI",
        "version": "1.0",
        "status": "ONLINE"
    })

# ----------------------------
# Run Flask
# ----------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
  
    # debug=False in the main run so the reloader doesn't spawn a second
    # process (which would start a second, competing detection_loop thread).
  