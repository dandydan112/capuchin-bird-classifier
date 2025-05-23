import sounddevice as sd
import numpy as np
import librosa
import joblib
import requests
import time
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo 

timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

model = joblib.load("capuchin_model.pkl")

# Læs konfigurationsfilen
with open('config.json') as config_file:
    config = json.load(config_file)

azure_url = config["azure_url"]

CONFIDENCE_THRESHOLD = 0.65

headers = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": "ff53d9578575498897e99d7e536e6464"
}

duration = 2
sample_rate = 22050
device_index = 1

def extract_mfcc(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

print("Listening for Capuchin sounds... (Press Ctrl+C to stop)")

try:
    while True:
        print("Recording...")
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            device=device_index
        )
        sd.wait()

        y = recording.flatten()
        y = librosa.util.normalize(y)
        features = extract_mfcc(y, sample_rate)

        confidence = model.predict_proba([features])[0][1]

        if confidence >= CONFIDENCE_THRESHOLD:
            print(f"Capuchin detected! Confidence: {confidence:.2f}")

            
            data = {
                "detected": True,
                "timestamp": timestamp,
                "confidence": float(confidence),
                "model_version": "v1.0.0",
                "recordingURL": "Ingen URL",
                "location": {
                    "lat": 55.4245,
                    "lon": 10.4495
                    }
            }

            try:
                response = requests.post(azure_url, json=data, headers=headers, timeout=5)
                response.raise_for_status()
                print("Notification sent to Azure:", response.status_code)
            except requests.RequestException as e:
                print("Failed to notify Azure:", e)
        else:
            print(f"No Capuchin detected. Confidence: {confidence:.2f}")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopped.")