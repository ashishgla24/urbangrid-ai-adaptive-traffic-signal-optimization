from flask import Flask, render_template
import pandas as pd
import os
from incident_detector import analyze_incidents

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "traffic_dataset.csv")

@app.route("/")
def dashboard():

    lane1 = lane2 = lane3 = lane4 = total = prediction = 0
    congestion = "LOW"
    signal = "GREEN"

    chart_labels = []
    chart_values = []

    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)

        latest = df.iloc[-1]

        lane1 = int(latest["lane1"])
        lane2 = int(latest["lane2"])
        lane3 = int(latest["lane3"])
        lane4 = int(latest["lane4"])

        total = int(latest["total_traffic"])

        chart_labels = df["timestamp"].tolist()
        chart_values = df["total_traffic"].tolist()

        if total > 30:
            congestion = "HIGH"
        elif total > 15:
            congestion = "MEDIUM"

        signal = "GREEN" if lane2 == max(lane1, lane2, lane3, lane4) else "RED"

        prediction = total + 2

    incident = analyze_incidents()

    return render_template(
        "index.html",
        lane1=lane1,
        lane2=lane2,
        lane3=lane3,
        lane4=lane4,
        total=total,
        prediction=prediction,
        congestion=congestion,
        signal=signal,
        chart_labels=chart_labels,
        chart_values=chart_values,
        incident_status=incident["status"],
        incident_alert=incident["alert"],
        incident_lane=incident["lane"],
        incident_action=incident["action"],
    )

if __name__ == "__main__":
    app.run(debug=True)