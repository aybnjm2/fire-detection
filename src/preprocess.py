import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from src.config import SCALER_PATH

def preprocess_data(df, save_scaler=True):
    # 1. Clean Data
    df = df.dropna(subset=['brightness', 'frp', 'type']).copy()
    
    # 2. Categorical Encodings
    df['daynight'] = df['daynight'].map({'D': 1, 'N': 0})
    
    df = pd.get_dummies(df, columns=['satellite'], dtype=float)
    if 'satellite_Aqua' not in df.columns: df['satellite_Aqua'] = 0.0
    if 'satellite_Terra' not in df.columns: df['satellite_Terra'] = 0.0

    # 3. Temporal Features
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    df['month'] = df['acq_date'].dt.month
    df['day_of_week'] = df['acq_date'].dt.dayofweek
    df['day_of_year'] = df['acq_date'].dt.dayofyear

    # Safely handle time parsing
    if 'acq_time' in df.columns:
        df['time_minutes'] = (df['acq_time'] // 100) * 60 + (df['acq_time'] % 100)
    else:
        # Fallback if only acq_date is provided (e.g., synthetic data)
        df['time_minutes'] = df['acq_date'].dt.hour * 60 + df['acq_date'].dt.minute
    
    # 4. Scale features
    num_cols = ['brightness', 'bright_t31', 'frp']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Save the scaler for inference
    if save_scaler:
        joblib.dump(scaler, SCALER_PATH)
        
    return df, scaler