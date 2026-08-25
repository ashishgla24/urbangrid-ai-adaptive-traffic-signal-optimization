from flask import Flask, render_template, jsonify, Response, send_file
import os
import time
import threading
import cv2
from ultralytics import YOLO
from .incident_detector import analyze_incidents
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from urbangrid.emergency_detector import EmergencyVehicleDetector
app = Flask(__name__)

# ----------------------------
# Paths
# ----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "raw", "traffic.mp4")
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "traffic_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "yolov8s.pt")

# ----------------------------
# Load YOLO model
# ----------------------------

model = YOLO(MODEL_PATH)

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

            # model.track() keeps a persistent ID per vehicle across frames
            # via ByteTrack, so a car sitting at a red light for 50 frames
            # is one vehicle, not 50 detections. conf/iou are raised from
            # the ultralytics defaults (~0.25 conf) to cut false positives
            # and duplicate boxes in dense, overlapping traffic. Falls back
            # to plain model.predict() if the tracker deps aren't installed.
            if tracking_available:
                try:
                    results = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=0.10,
                        iou=0.45,
                        verbose=False
                    )[0]
                except Exception as track_err:
                    print(
                        "[detection_loop] tracking unavailable "
                        f"({track_err}); falling back to plain detection. "
                        "Install 'lapx' (pip install lapx) to enable "
                        "per-vehicle tracking."
                    )
                    tracking_available = False
                    results = model.predict(
                        frame, conf=0.10, iou=0.45, verbose=False
                    )[0]
            else:
                results = model.predict(
                    frame, conf=0.10, iou=0.45, verbose=False
                )[0]
                vehicle_boxes = 0
                if results.boxes is not None:
                    for b in results.boxes:
                      if int(b.cls[0]) in vehicle_classes:
                         vehicle_boxes += 1
                         print("Detected vehicles:", vehicle_boxes,flush=True)

            h, w, _ = frame.shape
            # Draw lane boundaries
            cv2.line(frame, (int(w*0.25), 0), (int(w*0.25), h), (255,255,0), 2)
            cv2.line(frame, (int(w*0.50), 0), (int(w*0.50), h), (255,255,0), 2)
            cv2.line(frame, (int(w*0.75), 0), (int(w*0.75), h), (255,255,0), 2)
            lane_counts = [0, 0, 0, 0]
            vehicle_count = 0
            now = time.time()
            boxes = results.boxes
            has_ids = boxes is not None and getattr(boxes, "id", None) is not None
            emergency_input = []
            if boxes is not None:
                classes = boxes.cls.int().tolist()
                xyxy = boxes.xyxy.tolist()

                if has_ids:
                    ids = boxes.id.int().tolist()
                else:
                    # No tracker IDs available - assign throwaway per-frame
                    # IDs just so the label matches the on-screen style;
                    # these are NOT persistent, so throughput below only
                    # counts them as "seen right now", not deduplicated.
                    ids = list(range(next_fallback_id, next_fallback_id + len(classes)))
                    next_fallback_id += len(classes)

                for track_id, cls, (x1, y1, x2, y2) in zip(ids, classes, xyxy):
                    if cls not in vehicle_classes:
                       continue

                    vehicle_count += 1
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    center_x = (x1 + x2) // 2

                   # Temporary lane boundaries
                    lane1_end = int(w * 0.25)
                    lane2_end = int(w * 0.50)
                    lane3_end = int(w * 0.75)

                    if center_x < lane1_end:
                      lane = 0
                    elif center_x < lane2_end:
                       lane = 1
                    elif center_x < lane3_end:
                       lane = 2
                    else:
                      lane = 3

                    lane_counts[lane] += 1
                    emergency_input.append({
                       "label": model.names[cls],
                       "bbox": (x1, y1, x2, y2)
                    }) 
                    if has_ids and track_id not in track_first_seen:
                        track_first_seen[track_id] = now

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{model.names[cls]} #{track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
        except Exception as loop_err:
            # Never let an unexpected error silently kill this thread -
            # that's exactly what made the dashboard "stop counting" with
            # no explanation before. Log it, skip this frame, keep going.
            print(f"[detection_loop] frame skipped due to error: {loop_err}")
            time.sleep(0.1)
            continue

        # drop track IDs we haven't seen in a while so this dict doesn't
        # grow forever over a long-running stream
        if now - last_prune > 10:
            cutoff = now - (THROUGHPUT_WINDOW_SEC * 2)
            track_first_seen = {
                tid: ts for tid, ts in track_first_seen.items() if ts >= cutoff
            }
            last_prune = now

        # Detect emergency vehicle
        emergency_info = emergency_detector.detect(
            emergency_input,
            w
        )

        frame = emergency_detector.draw(frame, emergency_info)

        total = vehicle_count

        if has_ids:
            # real "vehicles/minute": unique vehicles first seen in the
            # last 60 seconds, not total*60 extrapolated from one frame
            vehicles_per_minute = sum(
                1 for ts in track_first_seen.values()
                if now - ts <= THROUGHPUT_WINDOW_SEC
            )
        else:
            # tracking unavailable this frame - fall back to the rough
            # extrapolation rather than showing 0
            vehicles_per_minute = total * 60

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
    incident = analyze_incidents()

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
        incident_status=incident["status"],
        incident_alert=incident["alert"],
        incident_lane=incident["lane"],
        incident_action=incident["action"]
    )


# ----------------------------
# Live API
# ----------------------------

@app.route("/api/live")
def live_api():
    incident = analyze_incidents()

    with data_lock:
        data = dict(live_data)
        data["chart_labels"] = list(chart_labels)
        data["chart_values"] = list(chart_values)

    data.update({
        "incident_status": incident["status"],
        "incident_alert": incident["alert"],
        "incident_lane": incident["lane"],
        "incident_action": incident["action"],
    })

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
    threading.Thread(target=detection_loop, daemon=True).start()
    # debug=False in the main run so the reloader doesn't spawn a second
    # process (which would start a second, competing detection_loop thread).
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)