import requests
import time
import os

ESP32_IP = os.getenv("ESP32_IP", "10.10.0.2")
ESP32_PORT = os.getenv("ESP32_PORT", "80")


def send_signal(signal):
    signal = signal.lower()

    url = f"http://{ESP32_IP}:{ESP32_PORT}/{signal}"

    try:
        response = requests.get(url, timeout=3)

        if response.status_code == 200:
            print(f"ESP32 SIGNAL: {signal.upper()} -> OK")
            print("ESP32 RESPONSE:", response.text)
            return True

        print(f"ESP32 ERROR: HTTP {response.status_code}")
        return False

    except requests.RequestException as e:
        print(f"ESP32 CONNECTION FAILED: {e}")
        return False


if __name__ == "__main__":

    print("UrbanGrid AI ESP32 Controller")

    while True:

        send_signal("red")
        time.sleep(3)

        send_signal("yellow")
        time.sleep(2)

        send_signal("green")
        time.sleep(5)