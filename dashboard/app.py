import os
from flask import Flask, jsonify, render_template, request, url_for
from incident_detector import analyze_incidents
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "traffic_dataset.csv")


def get_latest_data():
  if not os.path.exists(DATASET_PATH):
    return {
        "lane1": 0,
        "lane2": 0,
        "lane3": 0,
        "lane4": 0,
        "total": 0,
        "prediction": 0,
        "congestion": "LOW",
        "signal": "RED",
        "chart_labels": [],
        "chart_values": [],
        "fps": 12.5,
        "throughput": 0,
        "congestion_score": 0,
        "system_health": "HEALTHY",
    }
  df = pd.read_csv(DATASET_PATH)
  if len(df) == 0:
    return {
        "lane1": 0,
        "lane2": 0,
        "lane3": 0,
        "lane4": 0,
        "total": 0,
        "prediction": 0,
        "congestion": "LOW",
        "signal": "RED",
        "chart_labels": [],
        "chart_values": [],
        "fps": 12.5,
        "throughput": 0,
        "congestion_score": 0,
        "system_health": "HEALTHY",
    }
  latest = df.iloc[-1]
  lane1 = int(latest["lane1"])
  lane2 = int(latest["lane2"])
  lane3 = int(latest["lane3"])
  lane4 = int(latest["lane4"])
  total = int(latest["total_traffic"])

  if total > 30:
    congestion = "HIGH"
  elif total > 15:
    congestion = "MEDIUM"
  else:
    congestion = "LOW"

  lanes = [lane1, lane2, lane3, lane4]
  max_lane = lanes.index(max(lanes)) + 1
  signal = f"GREEN - Lane {max_lane}"
  prediction = total + 2
  fps = 12.5
  throughput = total * 60
  congestion_score = min(total * 2, 100)

  return {
      "lane1": lane1,
      "lane2": lane2,
      "lane3": lane3,
      "lane4": lane4,
      "total": total,
      "prediction": prediction,
      "congestion": congestion,
      "signal": signal,
      "chart_labels": df["timestamp"].tolist(),
      "chart_values": df["total_traffic"].tolist(),
      "fps": fps,
      "throughput": throughput,
      "congestion_score": congestion_score,
      "system_health": "HEALTHY",
  }


@app.route("/")
def dashboard():
  data = get_latest_data()
  incident = analyze_incidents()
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
      chart_labels=data["chart_labels"],
      chart_values=data["chart_values"],
      fps=data["fps"],
      throughput=data["throughput"],
      congestion_score=data["congestion_score"],
      incident_status=incident["status"],
      incident_alert=incident["alert"],
      incident_lane=incident["lane"],
      incident_action=incident["action"],
  )


@app.route("/api/live")
def live_api():
  data = get_latest_data()
  incident = analyze_incidents()
  data.update({
      "incident_status": incident["status"],
      "incident_alert": incident["alert"],
      "incident_lane": incident["lane"],
      "incident_action": incident["action"],
  })
  return jsonify(data)


@app.route("/api/status")
def system_status():
  return jsonify({
      "system": "UrbanGrid AI",
      "version": "1.0",
      "status": "ONLINE",
      "modules": [
          "YOLOv8 Detection",
          "Vehicle Tracking",
          "Lane Density Analysis",
          "Adaptive Signal Control",
          "Emergency Vehicle Priority",
          "LSTM Forecasting",
          "Incident Detection",
          "Live Dashboard",
      ],
  })


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)
