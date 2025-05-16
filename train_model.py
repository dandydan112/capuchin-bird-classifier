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

# Feature extractor
def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)


# Step 1: Prepare training data
X, y = [], []

print("Extracting Capuchin features...")
for fname in os.listdir(capuchin_dir):
    if fname.endswith(".wav"):
        X.append(extract_features(os.path.join(capuchin_dir, fname)))
        y.append(1)

print("Extracting Non-Capuchin features...")
for fname in os.listdir(non_capuchin_dir):
    if fname.endswith(".wav"):
        X.append(extract_features(os.path.join(non_capuchin_dir, fname)))
        y.append(0)

X, y = np.array(X), np.array(y)

# Step 2: Train classifier
print("Training classifier...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

print("Evaluation:")
print(classification_report(y_test, clf.predict(X_test)))

# Save model
joblib.dump(clf, "capuchin_model.pkl")
print("Model saved to capuchin_model.pkl")
