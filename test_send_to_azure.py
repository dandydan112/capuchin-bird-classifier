import requests
from datetime import datetime

url = ""

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
    response = requests.post(url, json=data, timeout=5)
    response.raise_for_status()
    print("Test request sent successfully:", response.status_code)
    print("Response content:", response.text)
except requests.RequestException as e:
    print("Failed to send test request:", e)