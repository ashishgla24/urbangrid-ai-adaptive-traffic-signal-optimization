import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "traffic_dataset.csv")

def analyze_incidents():

    if not os.path.exists(DATASET_PATH):
        return {
            "status": "NO DATA",
            "alert": "Traffic dataset not found",
            "lane": "N/A",
            "action": "Collect traffic data first"
        }

    df = pd.read_csv(DATASET_PATH)

    latest = df.iloc[-1]

    lane1 = int(latest["lane1"])
    lane2 = int(latest["lane2"])
    lane3 = int(latest["lane3"])
    lane4 = int(latest["lane4"])

    lanes = [lane1, lane2, lane3, lane4]

    max_lane = lanes.index(max(lanes)) + 1

    total = int(latest["total_traffic"])

    if total >= 30:
        status = "CRITICAL"
        alert = "Severe traffic congestion detected"
        action = "Activate emergency traffic diversion"

    elif total >= 20:
        status = "HIGH"
        alert = "Possible traffic incident or bottleneck"
        action = "Extend green signal timing"

    elif total >= 10:
        status = "MEDIUM"
        alert = "Traffic building up"
        action = "Monitor intersection continuously"

    else:
        status = "LOW"
        alert = "Traffic flowing normally"
        action = "Normal signal operation"

    return {
        "status": status,
        "alert": alert,
        "lane": f"Lane {max_lane}",
        "action": action
    }