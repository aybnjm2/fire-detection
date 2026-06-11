import numpy as np
import hdbscan
import joblib
from src.config import CLUSTERER_PATH

def apply_clustering(df, save_clusterer=True):
    # Strict Radian Conversion for Haversine
    spatial_coords = np.radians(df[['latitude', 'longitude']])
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15, 
        min_samples=5, 
        metric='haversine',
        prediction_data=True  # Allows future inference
    )
    
    df['cluster_id'] = clusterer.fit_predict(spatial_coords)
    
    # Save the clustering model for inference
    if save_clusterer:
        joblib.dump(clusterer, CLUSTERER_PATH)
        
    return df, clusterer