import pickle
import pandas as pd
import lightgbm as lgb
import os
import requests
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE

# URL GitHub Releases
BASE_URL = "https://github.com/injanisgg/wqi-classify/releases/download/v1.0.0/"
FILES = ["lgbm_pipeline_model.pkl", "rfe.pkl", "scaler.pkl", "selected_features.pkl"]
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

# Load dataset
data = pd.read_csv('waterQuality.csv')

# Debugging: Print original column names
# print("Original columns:", data.columns.tolist())

# Bersihkan dataset dari nilai '#NUM!'
if (data == '#NUM!').any().any():
    data = data.replace('#NUM!', float('nan'))
    data = data.dropna()

# Konversi semua kolom ke tipe numerik (antisipasi kolom non-numerik)
data = data.apply(pd.to_numeric, errors='coerce')

# Pisahkan fitur dan target
X = data.drop(columns=['is_safe'])
y = data['is_safe']

# Load model dan transformer dari file
with open(os.path.join(MODEL_DIR, "scaler.pkl"), 'rb') as f:
    final_scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "rfe.pkl"), 'rb') as f:
    rfe = pickle.load(f)

with open(os.path.join(MODEL_DIR, "selected_features.pkl"), 'rb') as f:
    selected_feature_names = pickle.load(f)

with open(os.path.join(MODEL_DIR, "lgbm_pipeline_model.pkl"), 'rb') as f:
    final_model = pickle.load(f)

# Terapkan preprocessing
X_scaled = final_scaler.transform(X[selected_feature_names])
X_selected_df = pd.DataFrame(X_scaled, columns=selected_feature_names)

# Model siap digunakan
print("Model dan preprocessing berhasil dimuat!")