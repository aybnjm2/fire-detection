import os

# Base directory mapping
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Paths
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Model Paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
CLUSTERER_PATH = os.path.join(MODELS_DIR, 'spatial_clusterer.pkl')
REG_MODEL_PATH = os.path.join(MODELS_DIR, 'reg_frp_model.keras')
CLS_MODEL_PATH = os.path.join(MODELS_DIR, 'cls_type_model.keras')