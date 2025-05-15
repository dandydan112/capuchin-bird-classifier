import sounddevice as sd
import numpy as np
import librosa
import joblib
import requests
import time
import RPi.GPIO as GPIO
from datetime import datetime
from zoneinfo import ZoneInfo
from gpiozero import LED, Buzzer
from gpiozero.pins.lgpio import LGPIOFactory
from gpiozero.devices import Device
from time import sleep

# Set up the GPIO pin factory to use LGPIO (works in virtual environments)
Device.pin_factory = LGPIOFactory()

# GPIO setup for LED and buzzer
red_led = LED(17)       # Red LED on GPIO 27 (Pin 13)
green_led = LED(27)     # Green LED on GPIO 17 (Pin 11)
buzzer = Buzzer(22)     # Buzzer on GPIO 22 (Pin 15)

# Default state
red_led.on()
green_led.off()
buzzer.off()

# Load model
model = joblib.load("capuchin_model.pkl")

azure_url = ""
CONFIDENCE_THRESHOLD = 0.70

headers = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": "ff53d9578575498897e99d7e536e6464"
}

duration = 2
sample_rate = 22050
device_index = 1

# Feature extraction function
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

            
            green_led.on()
            red_led.off()

            # Short buzzer pulse
            buzzer.on()
            time.sleep(0.2)
            buzzer.off()

            
            data = {
                "detected": True,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "confidence": float(confidence),
                "model_version": "v1.0.0",
                "recordingURL": "",
                "location": {
                    "lat": 55.6761,
                    "lon": 12.5683
                }
            }

            # try:
            #     response = requests.post(azure_url, json=data, headers=headers, timeout=5)
            #     response.raise_for_status()
            #     print("Notification sent to Azure:", response.status_code)
            # except requests.RequestException as e:
            #     print("Failed to notify Azure:", e)

        else:
            print(f"No Capuchin detected. Confidence: {confidence:.2f}")

            # Reset state
            green_led.off()
            red_led.on()

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    red_led.off()
    green_led.off()
    buzzer.off()
   
