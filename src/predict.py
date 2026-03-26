import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Fallback for relative imports
try:
    from .config import MODELS_DIR
    from .utils import load_model
except ImportError:
    MODELS_DIR = 'models'
    def load_model(name):
        return joblib.load(os.path.join(MODELS_DIR, name))

def get_prediction(inputs_dict, model_name='best_model.pkl'):
    """
    Transform input and predict using best model.
    """
    try:
        scaler = load_model('scaler.pkl')
        selector = load_model('selector.pkl')
        encoder = load_model('encoder.pkl')
        model = load_model('best_model.pkl')
        
        input_df = pd.DataFrame([inputs_dict])
        
        # Encode known cat cols
        cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status']
        for col in cat_cols:
            if col in input_df and col in encoder:
                input_df[col] = encoder[col].transform(input_df[col].astype(str))
        
        # Pad missing cols with 0 (assume order match)
        input_df = input_df.reindex(columns=selector.feature_names_in_ if hasattr(selector, 'feature_names_in_') else range(selector.n_features_in_), fill_value=0)
        
        X_input = selector.transform(input_df)
        X_scaled = scaler.transform(X_input)
        
        proba = model.predict_proba(X_scaled)[0]
        pred = model.predict(X_scaled)[0]
        
        risk_map = {0: 'Low Risk (0)', 1: 'Moderate Risk (1)', 2: 'High Risk (2)'}
        
        return pred, proba, risk_map[pred]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, "Train model first!"
