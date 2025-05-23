import os
import numpy as np
import librosa
import joblib

#PAth til folder som skal classificeres
forest_dir = r"data\Forest Recordings"
model_path = "capuchin_model.pkl"

#Feature extractor metoden
def extract_features(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

#Loader modellen
clf = joblib.load(model_path)
print("Model loaded.")

# Sliding windows
chunk_duration = 2.0
step_duration = 1.0

print("\nScanning forest recordings...\n")

for fname in os.listdir(forest_dir):
    if fname.endswith(".wav"):
        path = os.path.join(forest_dir, fname)
        y, sr = librosa.load(path, sr=22050)

        chunk_size = int(sr * chunk_duration)
        step_size = int(sr * step_duration)
        found = False

        for start in range(0, len(y) - chunk_size + 1, step_size):
            end = start + chunk_size
            features = extract_features(y[start:end], sr)
            pred = clf.predict([features])[0]

            if pred == 1:
                print(f"{fname}: Capuchin between {start / sr:.2f}s – {end / sr:.2f}s")
                found = True

        if not found:
            print(f"{fname}: No Capuchin detected.")

            