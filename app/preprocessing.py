import pickle
import pandas as pd
import lightgbm as lgb
import os
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE

# Pastikan direktori models ada
if not os.path.exists('app/models'):
    os.makedirs('app/models')

# Load dataset
data = pd.read_csv('waterQuality.csv')

# # Debugging: Print original column names
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

# Simpan nama kolom original untuk scaler
feature_names = X.columns.tolist()

# Terapkan RobustScaler pada fitur
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# Buat dan fit RFE dengan LightGBM
gbm = lgb.LGBMClassifier()
rfe = RFE(estimator=gbm, n_features_to_select=18, step=1)
rfe = rfe.fit(X_scaled_df, y)

# Dapatkan nama fitur yang terpilih
selected_features_mask = rfe.support_
selected_feature_names = X.columns[selected_features_mask].tolist()

# print("\nSelected features:", selected_feature_names)
# print("Number of selected features:", len(selected_feature_names))

# Reset scaler dengan hanya menggunakan fitur terpilih
final_scaler = RobustScaler()
final_scaler.fit(X[selected_feature_names])

# Simpan model dan transformers
with open('app/models/scaler.pkl', 'wb') as f:
    pickle.dump(final_scaler, f)

with open('app/models/rfe.pkl', 'wb') as f:
    pickle.dump(rfe, f)

with open('app/models/selected_features.pkl', 'wb') as f:
    pickle.dump(selected_feature_names, f)

# Train dan simpan model final
X_selected = X_scaled_df[selected_feature_names]
final_model = lgb.LGBMClassifier()
final_model.fit(X_selected, y)

with open('app/models/lgbm_pipeline_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)