import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os
import joblib

# Fix relative import issue
try:
    from .config import DATA_RAW, DATA_PROCESSED, TOP_FEATURES, FIGURES_DIR, MODELS_DIR
    from .utils import load_data, plot_distribution, plot_correlation
except (ImportError, ValueError):
    BASE_DIR = '.'
    DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw', 'mental_health_risk_dataset.csv')
    DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', 'clean_data.csv')
    TOP_FEATURES = 15
    FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    def load_data(path):
        return pd.read_csv(path)
    def plot_distribution(df, col):
        pass
    def plot_correlation(df_num):
        pass

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def detect_outliers(df, col):
    """Detect outliers using IQR."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (df[col] < lower) | (df[col] > upper)

def preprocess_data():
    """Complete preprocessing pipeline."""
    df = load_data(DATA_RAW)
    
    # Handle duplicates
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"Removed {initial_shape[0] - df.shape[0]} duplicates")
    
    # Missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())
        print(f"Handled {missing_before} missing values")
    
    # Outliers (cap to 1%/99%)
    num_cols = df.select_dtypes(np.number).columns
    for col in num_cols:
        lower, upper = df[col].quantile([0.01, 0.99])
        outliers = detect_outliers(df, col)
        df.loc[outliers, col] = np.clip(df[col][outliers], lower, upper)
    
    # Encode categoricals
    cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Split features/target
    X = df.drop('mental_health_risk', axis=1)
    y = df['mental_health_risk']
    
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=TOP_FEATURES)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Save processed
    df_processed = pd.DataFrame(X_scaled, columns=selected_features)
    df_processed['mental_health_risk'] = y.values
    df_processed.to_csv(DATA_PROCESSED, index=False)
    
# Save artifacts
    joblib.dump(encoders, os.path.join(MODELS_DIR, 'encoder.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(selector, os.path.join(MODELS_DIR, 'selector.pkl'))
    joblib.dump(selected_features, os.path.join(MODELS_DIR, 'selected_features.pkl'))
    
    print("Preprocessing complete!")
    return df_processed, selected_features, encoders, scaler, selector

if __name__ == '__main__':
    preprocess_data()
