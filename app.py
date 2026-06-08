from flask import Flask, render_template, request
import joblib
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = joblib.load("model/car_model.pkl")
encoder = joblib.load("model/label_encoder.pkl")

IMG_SIZE = 128


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "Tidak ada file"

    file = request.files["image"]

    if file.filename == "":
        return "File kosong"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(filepath)

    img = cv2.imread(filepath)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img.flatten()

    img = np.array([img])

    pred = model.predict(img)

    result = encoder.inverse_transform(pred)[0]

    return render_template(
        "index.html",
        prediction=result,
        image=file.filename
    )


if __name__ == "__main__":
    app.run(debug=True)