import requests
from datetime import datetime

url = "https://dronedetection-apim.azure-api.net/dronealarm-m/SaveDroneAlarm"

headers = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": "ff53d9578575498897e99d7e536e6464"
}

data = {
    "detected": True,
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "confidence": 0.97,
    "model_version": "v1.0.0",
    "recordingURL": "harikkeurltilmp3endnu.dk",
    "location": {
        "lat": 55.6761,
        "lon": 12.5683
    }
}

try:
    response = requests.post(url, json=data, headers=headers, timeout=15)
    response.raise_for_status()
    print("Test request sent successfully:", response.status_code)
    print("Response content:", response.text)
except requests.RequestException as e:
    print("Failed to send test request:", e)