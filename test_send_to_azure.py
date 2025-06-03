import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo 
import json

timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# Læs konfigurationsfilen
with open('config.json') as config_file:
    config = json.load(config_file)

azure_url = config["azure_url"]

headers = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": "ff53d9578575498897e99d7e536e6464"
}

data = {
    "detected": True,
    "timestamp": timestamp,
    "confidence": 0.79,
    "model_version": "v1.0.0",
    "recordingURL": "Ingen URL",
    "location": {
        "lat": 55.4662,
        "lon": 9.7624
    }
}


try:
    response = requests.post(azure_url, json=data, headers=headers, timeout=15)
    response.raise_for_status()
    print("Test request sent successfully:", response.status_code)
    print("Response content:", response.text)
except requests.RequestException as e:
    print("Failed to send test request:", e)