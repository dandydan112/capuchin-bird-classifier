
import sounddevice as sd
import numpy as np
import librosa
import joblib
import time

# Load trained model
model = joblib.load("capuchin_model.pkl")

# Audio settings
duration = 2  # seconds
sample_rate = 22050

def extract_mfcc(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

print("Listening for Capuchin sounds... (Ctrl+C for at stoppe)")

import sounddevice as sd

try:
    while True:
        print("Recording...")
        recording = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype='float32',
    device=1  
)

        sd.wait()

        y = recording.flatten()
        y = librosa.util.normalize(y)  # Normalisér volumen
        features = extract_mfcc(y, sample_rate)

        prediction = model.predict([features])[0]

        if prediction == 1:
            print("Capuchin sound detected!")
        else:
            print("No Capuchin detected.")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopped.")
