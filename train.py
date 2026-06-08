import os
import cv2
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

IMG_SIZE = 128

train_path = "dataset mobil/train"
test_path = "dataset mobil/test"

X_train = []
y_train = []

print("Loading training data...")

for label in os.listdir(train_path):

    folder = os.path.join(train_path, label)

    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):

        img_path = os.path.join(folder, file)

        try:
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            img = img.flatten()

            X_train.append(img)
            y_train.append(label)

        except:
            pass

print("Training images:", len(X_train))

encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# TEST

X_test = []
y_test = []

print("Loading test data...")

for label in os.listdir(test_path):

    folder = os.path.join(test_path, label)

    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):

        img_path = os.path.join(folder, file)

        try:
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            img = img.flatten()

            X_test.append(img)
            y_test.append(label)

        except:
            pass

y_test = encoder.transform(y_test)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f"Akurasi: {acc*100:.2f}%")

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/car_model.pkl")
joblib.dump(encoder, "model/label_encoder.pkl")

print("Model berhasil disimpan!")