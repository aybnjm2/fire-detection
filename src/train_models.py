import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from src.config import REG_MODEL_PATH, CLS_MODEL_PATH
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization

def build_and_train_regression(df, save_model=True):
    X_reg = df[['latitude', 'longitude', 'brightness', 'bright_t31', 
                'daynight', 'satellite_Aqua', 'satellite_Terra']].values
    y_reg = df['frp'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        BatchNormalization(), # stabiliser entrainement
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='huber', metrics=['mse'])

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # early stopping

    print("Training Regression Model (FRP)...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[es], verbose=0)
    
    if save_model:
        model.save(REG_MODEL_PATH)
        
    return model

def build_and_train_classification(df, trained_reg_model, save_model=True):
    print("\nCorrecting Covariate Shift: Predicting FRP for Classifier Training...")
    
    X_reg = df[['latitude', 'longitude', 'brightness', 'bright_t31', 
                'daynight', 'satellite_Aqua', 'satellite_Terra']].values
    
    predicted_frp = trained_reg_model.predict(X_reg, verbose=0).flatten()
    
    df_shifted = df.copy()
    df_shifted['predicted_frp'] = predicted_frp
    
    X_cls = df_shifted[['brightness', 'scan', 'track', 'confidence', 
                        'bright_t31', 'predicted_frp', 'daynight', 'cluster_id']].values
    
    y_cls = pd.get_dummies(df_shifted['type']).values
    num_classes = y_cls.shape[1]
    
    X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    
    y_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    weight_dict = dict(enumerate(class_weights))
    
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Training Classification Model (Fire Type) on Predicted FRP...")
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2, class_weight=weight_dict, verbose=0)
    
    if save_model:
        model.save(CLS_MODEL_PATH)
        
    return model