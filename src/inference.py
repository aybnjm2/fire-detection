import numpy as np
import pandas as pd
import hdbscan
import joblib
from tensorflow.keras.models import load_model
from src.config import SCALER_PATH, CLUSTERER_PATH, REG_MODEL_PATH, CLS_MODEL_PATH

def load_artifacts():
    scaler = joblib.load(SCALER_PATH)
    clusterer = joblib.load(CLUSTERER_PATH)
    reg_model = load_model(REG_MODEL_PATH)
    cls_model = load_model(CLS_MODEL_PATH)
    return scaler, clusterer, reg_model, cls_model

def predict_new_anomaly(new_data_dict):
    # Load previously trained models and scalers
    scaler, clusterer, reg_model, cls_model = load_artifacts()
    
    df_new = pd.DataFrame([new_data_dict])
    
    # 1. Preprocess
    df_new['daynight'] = 1 if df_new['daynight'].iloc[0] == 'D' else 0
    df_new['satellite_Aqua'] = 1.0 if df_new['satellite'].iloc[0] == 'Aqua' else 0.0
    df_new['satellite_Terra'] = 1.0 if df_new['satellite'].iloc[0] == 'Terra' else 0.0
    
    df_new['frp'] = 0 # Dummy value
    df_new[['brightness', 'bright_t31', 'frp']] = scaler.transform(df_new[['brightness', 'bright_t31', 'frp']])
    
    # 2. HDBSCAN Inference
    new_coords = np.radians(df_new[['latitude', 'longitude']])
    cluster_labels, _ = hdbscan.approximate_predict(clusterer, new_coords)
    df_new['cluster_id'] = cluster_labels[0]
    
    # 3. Regression Inference
    reg_features = df_new[['latitude', 'longitude', 'brightness', 'bright_t31', 'daynight', 'satellite_Aqua', 'satellite_Terra']].values
    predicted_frp_scaled = reg_model.predict(reg_features, verbose=0)[0][0]
    df_new['predicted_frp'] = predicted_frp_scaled 
    
    # 4. Classification Inference
    cls_features = df_new[['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'predicted_frp', 'daynight', 'cluster_id']].values
    type_probs = cls_model.predict(cls_features, verbose=0)[0]
    predicted_type = np.argmax(type_probs)
    
    # Inverse transform FRP to Real MW Value
    dummy_inverse = np.zeros((1, 3))
    dummy_inverse[0, 2] = predicted_frp_scaled
    real_frp = scaler.inverse_transform(dummy_inverse)[0, 2]
    
    return {
        "Predicted_FRP_MW": round(real_frp, 2),
        "Assigned_Cluster_ID": cluster_labels[0],
        "Predicted_Fire_Type": predicted_type,
        "Type_Confidence": round(type_probs[predicted_type] * 100, 2)
    }