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
            exit(1)  # Berhenti jika ada file yang gagal diunduh

# Load dataset dengan error handling
try:
    data = pd.read_csv('waterQuality.csv')
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    exit(1)

# Debugging: Print original column names
# print("Original columns:", data.columns.tolist())

# Bersihkan dataset dari nilai '#NUM!' dan pastikan semua nilai numerik
data.replace('#NUM!', float('nan'), inplace=True)
data.dropna(inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')

# Pisahkan fitur dan target
if 'is_safe' not in data.columns:
    print("❌ Error: 'is_safe' column not found in dataset!")
    exit(1)

X = data.drop(columns=['is_safe'])
y = data['is_safe']

# Load model dan transformer dari file dengan error handling
def load_pickle(file_name):
    file_path = os.path.join(MODEL_DIR, file_name)
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_name}: {e}")
        exit(1)

final_scaler = load_pickle("scaler.pkl")
rfe = load_pickle("rfe.pkl")
selected_feature_names = load_pickle("selected_features.pkl")
final_model = load_pickle("lgbm_pipeline_model.pkl")

# Terapkan preprocessing
X_scaled = final_scaler.transform(X[selected_feature_names])
X_selected_df = pd.DataFrame(X_scaled, columns=selected_feature_names)

# Model siap digunakan
print("✅ Model dan preprocessing berhasil dimuat!")