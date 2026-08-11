from flask import Flask, render_template, jsonify, Response, send_file
import os
import cv2
from ultralytics import YOLO
from incident_detector import analyze_incidents

app = Flask(__name__)

# ----------------------------

# Paths

# ----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "raw", "traffic.mp4")
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "traffic_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

# ----------------------------

# Load YOLO model

# ----------------------------

model = YOLO(MODEL_PATH)

# ----------------------------

# Global live traffic data

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
"fps": 12.5,
"throughput": 0,
"congestion_score": 0,
"system_health": "HEALTHY"
}

chart_labels = []
chart_values = []

# ----------------------------

# Video stream with live YOLO detection

# ----------------------------

def generate_frames():
    global live_data, chart_labels, chart_values

    cap = cv2.VideoCapture(VIDEO_PATH)
    while True:
        success, frame = cap.read()

        if not success:
            # restart video on end or wait briefly on read failure
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (960, 540))

        results = model(frame)[0]

        h, w, _ = frame.shape
        lane_counts = [0, 0, 0, 0]

        vehicle_classes = {2, 3, 5, 7}

        for box in results.boxes:
            cls = int(box.cls[0])

            if cls in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_x = (x1 + x2) // 2
                lane = min(center_x * 4 // w, 3)
                lane_counts[lane] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    model.names[cls],
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        total = sum(lane_counts)

        if total > 30:
            congestion = "HIGH"
        elif total > 15:
            congestion = "MEDIUM"
        else:
            congestion = "LOW"

        max_lane = lane_counts.index(max(lane_counts)) + 1
        signal = f"GREEN - Lane {max_lane}"

        live_data["lane1"] = lane_counts[0]
        live_data["lane2"] = lane_counts[1]
        live_data["lane3"] = lane_counts[2]
        live_data["lane4"] = lane_counts[3]
        live_data["total"] = total
        live_data["prediction"] = total + 2
        live_data["congestion"] = congestion
        live_data["signal"] = signal
        live_data["throughput"] = total * 60
        live_data["congestion_score"] = min(total * 2, 100)

        chart_labels.append(str(len(chart_labels)))
        chart_values.append(total)

        if len(chart_labels) > 20:
            chart_labels = chart_labels[-20:]
            chart_values = chart_values[-20:]

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )


# ----------------------------

# Dashboard

# ----------------------------

@app.route("/")
def dashboard():
    incident = analyze_incidents()

    return render_template(
    "index.html",
    lane1=live_data["lane1"],
    lane2=live_data["lane2"],
    lane3=live_data["lane3"],
    lane4=live_data["lane4"],
    total=live_data["total"],
    prediction=live_data["prediction"],
    congestion=live_data["congestion"],
    signal=live_data["signal"],
    chart_labels=chart_labels,
    chart_values=chart_values,
    fps=live_data["fps"],
    throughput=live_data["throughput"],
    congestion_score=live_data["congestion_score"],
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

    data = dict(live_data)
    data.update({
        "incident_status": incident["status"],
        "incident_alert": incident["alert"],
        "incident_lane": incident["lane"],
        "incident_action": incident["action"],
        "chart_labels": chart_labels,
        "chart_values": chart_values
    })

    return jsonify(data)


# ----------------------------

# Video stream

# ----------------------------

@app.route("/video_feed")
def video_feed():
        return Response(
                generate_frames(),
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
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
