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
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model jika belum ada
for file in FILES:
    file_path = os.path.join(MODEL_DIR, file)
    if not os.path.exists(file_path):
        print(f"Downloading {file}...")
        response = requests.get(BASE_URL + file)

        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"✅ {file} downloaded successfully!")
        else:
            print(f"❌ Failed to download {file}, status code {response.status_code}")
            exit(1)

# Fungsi untuk load model dengan error handling
def load_pickle(file_name):
    file_path = os.path.join(MODEL_DIR, file_name)
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_name}: {e}")
        exit(1)

# Load model dan preprocessor
model = load_pickle("lgbm_pipeline_model.pkl")
scaler = load_pickle("scaler.pkl")
selected_features = load_pickle("selected_features.pkl")

print("✅ Model and preprocessors loaded successfully!")

# Tentukan folder template
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Route index
@app.route("/")
def index():
    return render_template('index.html')

# Route model
@app.route("/model")
def render_model_page():
    return render_template('model.html')

# Route classification
@app.route("/classify", methods=['GET', 'POST'])
def classify():
    return render_template('classification.html')

# Route about
@app.route("/about")
def about():
    return render_template('about.html')

# Endpoint prediksi
@app.route("/classify/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        print("📥 Received data:", data)

        # Pastikan semua fitur tersedia dalam request
        missing_features = [feature for feature in selected_features if feature not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400

        # Urutkan data sesuai dengan fitur model
        input_data = [data[feature] for feature in selected_features]

        # Skalakan data
        scaled_data = scaler.transform([input_data])

        # Prediksi
        prediction = model.predict(scaled_data)

        # Konversi hasil prediksi menjadi label "Safe" atau "Not Safe"
        prediction_label = 'Safe' if int(prediction[0]) == 1 else 'Not Safe'

        print(f"🔍 Prediction: {prediction_label}")

        return jsonify({'prediction': prediction_label})

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mendapatkan fitur yang digunakan oleh model
@app.route('/classify/features', methods=['GET'])
def get_features():
    return jsonify({'selected_features': selected_features})

if __name__ == '__main__':
    app.run(debug=True)