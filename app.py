import pickle
import os
import requests
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

# URL GitHub Releases
BASE_URL = "https://github.com/injanisgg/wqi-classify/releases/download/v1.0.0/"
FILES = ["lgbm_pipeline_model.pkl", "scaler.pkl", "selected_features.pkl"]
MODEL_DIR = "app/models"

# Pastikan direktori models ada
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Download model jika belum ada
for file in FILES:
    file_path = os.path.join(MODEL_DIR, file)
    if not os.path.exists(file_path):
        print(f"Downloading {file}...")
        response = requests.get(BASE_URL + file)
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"{file} downloaded!")

# Load all required models and transformers
with open(os.path.join(MODEL_DIR, "lgbm_pipeline_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "selected_features.pkl"), "rb") as f:
    selected_features = pickle.load(f)

# print("Loaded selected features:", selected_features)

# Tentukan folder template
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# route index
@app.route("/")
def index():
    return render_template('index.html')

# route model
@app.route("/model")
def render_model_page():
    return render_template('model.html')

# route classification
@app.route("/classify", methods=['GET', 'POST'])
def classify():
    return render_template('classification.html')

# route about
@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/classify/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        # print("Received data:", data)

        # Urutkan data sesuai dengan fitur model
        input_data = [data[feature] for feature in selected_features]

        # Skalakan data
        scaled_data = scaler.transform([input_data])

        # Prediksi
        prediction = model.predict(scaled_data)
        # print("Prediction:", prediction)  # Debugging log untuk hasil prediksi

        # Konversi hasil prediksi menjadi label "Safe" atau "Not Safe"
        prediction_label = 'Safe' if int(prediction[0]) == 1 else 'Not Safe'

        return jsonify({'prediction': prediction_label})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/classify/features', methods=['GET'])
def get_features():
    return jsonify({'selected_features': selected_features})

if __name__ == '__main__':
    app.run(debug=True)