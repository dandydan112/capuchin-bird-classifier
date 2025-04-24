# Capuchin Bird Classifier
import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Folder paths
capuchin_dir = r"data\Parsed_Capuchinbird_Clips"
non_capuchin_dir = r"data\Parsed_Not_Capuchinbird_Clips"
forest_dir = r"data\Forest Recordings"

# Feature extractor
def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# Step 1: Prepare training data
X = []
y = []

print("Extracting MFCCs from Capuchin clips...")
for fname in os.listdir(capuchin_dir):
    if fname.endswith(".wav"):
        features = extract_features(os.path.join(capuchin_dir, fname))
        X.append(features)
        y.append(1)

print("Extracting MFCCs from Non-Capuchin clips...")
for fname in os.listdir(non_capuchin_dir):
    if fname.endswith(".wav"):
        features = extract_features(os.path.join(non_capuchin_dir, fname))
        X.append(features)
        y.append(0)

X = np.array(X)
y = np.array(y)

# Step 2: Train classifier
print("Training classifier...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

print("Evaluation:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
model_path = "capuchin_model.pkl"
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")

# Step 3: Predict forest audio
print("\nScanning forest clips...\n")
forest_results = []
for fname in os.listdir(forest_dir):
    if fname.endswith(".wav"):
        path = os.path.join(forest_dir, fname)
        features = extract_features(path)
        prediction = clf.predict([features])[0]
        if prediction == 1:
            print(f"Capuchin likely detected in: {fname}")
            forest_results.append(fname)

if not forest_results:
    print("No Capuchin calls detected in forest audio.")
