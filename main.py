import os
import pandas as pd
from src.config import MODELS_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR
from src.preprocess import preprocess_data
from src.clustering import apply_clustering
from src.train_models import build_and_train_regression, build_and_train_classification
from src.inference import predict_new_anomaly

def create_directories():
    """Ensure all required folders exist."""
    for directory in [MODELS_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR]:
        os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print("Project directories verified.")

    print("Loading raw data...")
    raw_df = pd.read_csv(os.path.join(DATA_RAW_DIR, 'MODIS_fires.csv'))

    print("\n--- PHASE 1: Preprocessing ---")
    processed_df, _ = preprocess_data(raw_df)

    print("\n--- PHASE 2: Clustering ---")
    clustered_df, _ = apply_clustering(processed_df)

    print("\n--- PHASE 3 & 4: Model Training ---")
    reg_model = build_and_train_regression(clustered_df)
    cls_model = build_and_train_classification(clustered_df, reg_model)

    print("\nAll models trained and saved to /models successfully.")

    print("\n--- PHASE 5: Live Inference Simulation ---")
    new_anomaly = {
        'latitude': 36.74, 'longitude': 36.21, 'brightness': 305.0, 
        'scan': 1.1, 'track': 1.0, 'confidence': 70, 
        'bright_t31': 281.5, 'daynight': 'N', 'satellite': 'Terra'
    }

    result = predict_new_anomaly(new_anomaly)
    print("Pipeline Output Profile:\n", result)