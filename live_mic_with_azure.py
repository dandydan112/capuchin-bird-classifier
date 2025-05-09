import sounddevice as sd
import numpy as np
import librosa
import joblib
import requests
import time
from datetime import datetime
from zoneinfo import ZoneInfo 

# Load trained model
model = joblib.load("capuchin_model.pkl")

# Maliks endpoint
azure_url = ""

# Audio settings
duration = 2  # seconds
sample_rate = 22050
device_index = 1  #mic 1

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

        # Predict
        prediction = model.predict([features])[0]
        confidence = model.predict_proba([features])[0][1]  # konfidens

        if prediction == 1:
            timestamp = datetime.now(ZoneInfo("Europe/Copenhagen")).isoformat()
            print(f"Capuchin detected! Confidence: {confidence:.2f}")

            data = {
                "detected": True,
                "timestamp": timestamp,
                "confidence": float(confidence),
                "model_version": "v1.0.0",
                "recordingURL": "",
                "location": {
                    "lat": 55.6761,
                    "lon": 12.5683
                }
            }

            try:
                response = requests.post(azure_url, json=data, timeout=5)
                response.raise_for_status()
                print("Notification sent to Azure:", response.status_code)
            except requests.RequestException as e:
                print("Failed to notify Azure:", e)
        else:
            print("No Capuchin detected.")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopped.")