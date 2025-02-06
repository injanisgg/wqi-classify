import pickle
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

# Load all required models and transformers
with open("app/models/lgbm_pipeline_model.pkl", "rb") as f:
    model = pickle.load(f)

with open('app/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('app/models/selected_features.pkl', 'rb') as f:
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
        selected_features = [
            "aluminium", "ammonia", "arsenic", "barium", "cadmium", 
            "chloramine", "chromium", "copper", "bacteria", "viruses", 
            "lead", "nitrates", "nitrites", "perchlorate", "radium", 
            "selenium", "silver", "uranium"
        ]
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