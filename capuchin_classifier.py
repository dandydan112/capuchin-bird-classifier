# Capuchin Bird Classifier
import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


capuchin_dir = r"data\Parsed_Capuchinbird_Clips"
non_capuchin_dir = r"data\Parsed_Not_Capuchinbird_Clips"
forest_dir = r"data\Forest Recordings"


def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)


X = []
y = []

print("Extracting MFCCs from Capuchin clips")
for fname in os.listdir(capuchin_dir):
    if fname.endswith(".wav"):
        features = extract_features(os.path.join(capuchin_dir, fname))
        X.append(features)
        y.append(1)

print("Extracting MFCCs from Non-Capuchin clips")
for fname in os.listdir(non_capuchin_dir):
    if fname.endswith(".wav"):
        features = extract_features(os.path.join(non_capuchin_dir, fname))
        X.append(features)
        y.append(0)

X = np.array(X)
y = np.array(y)


print("Training classifier")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

print("Evaluation:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


model_path = "capuchin_model.pkl"
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")


# Step 3: Predict forest audio
print("\n Scanning forest clips with sliding windows...\n")

chunk_duration = 2.0  # seconds
step_duration = 1.0   # Jeg overlapper med 1 sek


for fname in os.listdir(forest_dir):
    if fname.endswith(".wav"):
        file_path = os.path.join(forest_dir, fname)

        # Load full audio
        y, sr = librosa.load(file_path, sr=22050)
        total_duration = librosa.get_duration(y=y, sr=sr)

        chunk_size = int(sr * chunk_duration)
        step_size = int(sr * step_duration)

        found_capuchin = False

        for start in range(0, len(y) - chunk_size + 1, step_size):
            end = start + chunk_size
            chunk = y[start:end]

            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
            features = np.mean(mfcc.T, axis=0)

            prediction = clf.predict([features])[0]
            if prediction == 1:
                timestamp_start = start / sr
                timestamp_end = end / sr
                print(f"Capuchin detected in {fname} between {timestamp_start:.2f}s – {timestamp_end:.2f}s")
                found_capuchin = True

        if not found_capuchin:
            print(f"No Capuchin found in {fname}")