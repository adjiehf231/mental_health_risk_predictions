import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os

# Fallback imports for standalone/script mode
try:
    from .config import MODELS_DIR, METRICS_FILE, FIGURES_DIR, RANDOM_STATE, KFOLD, TOP_FEATURES
    from .utils import save_model, save_metrics, plot_roc_multi
    from .preprocessing import preprocess_data
except ImportError:
    BASE_DIR = '.'
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    METRICS_FILE = os.path.join(BASE_DIR, 'reports', 'metrics.json')
    FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
    RANDOM_STATE = 42
    KFOLD = 5
    TOP_FEATURES = 15
    def save_model(model, name):
        joblib.dump(model, os.path.join(MODELS_DIR, name))
    def save_metrics(metrics):
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f)
    def plot_roc_multi(y_test, y_proba):
        pass  # Skip ROC in fallback
    def preprocess_data():
        from preprocessing import preprocess_data
        return preprocess_data()

os.makedirs(FIGURES_DIR, exist_ok=True)

MODELS_DICT = {
    'C4.5 (DT)': DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=10),
    'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
}

def kfold_cv_scores(model, X, y):
    skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE)
    acc, prec, rec, f1, aucs = [], [], [], [], []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        
        acc.append(accuracy_score(y_val, y_pred))
        prec.append(precision_score(y_val, y_pred, average='macro', zero_division=0))
        rec.append(recall_score(y_val, y_pred, average='macro', zero_division=0))
        f1.append(f1_score(y_val, y_pred, average='macro', zero_division=0))
        aucs.append(roc_auc_score(y_val, y_proba, multi_class='ovr'))
    
    return {
        'Accuracy': np.mean(acc),
        'Precision': np.mean(prec),
        'Recall': np.mean(rec),
        'F1-Score': np.mean(f1),
        'ROC-AUC': np.mean(aucs)
    }

def train_models():
    # Preprocess full
    df_proc, selected_features, encoders, scaler, selector = preprocess_data()
    X = df_proc[selected_features]
    y = df_proc['mental_health_risk']
    
    metrics = {}
    for name, model in MODELS_DICT.items():
        print(f"K-fold CV for {name}...")
        scores = kfold_cv_scores(model, X, y)
        metrics[name] = scores
    
    metrics_df = pd.DataFrame(metrics).T
    reports_path = os.path.join(os.path.dirname(__file__), '..', 'reports')
    os.makedirs(reports_path, exist_ok=True)
    metrics_df.to_csv(os.path.join(reports_path, 'model_comparison.csv'))
    
    best_model_name = metrics_df['F1-Score'].idxmax()
    print(f"Best: {best_model_name} (F1: {metrics_df['F1-Score'][best_model_name]:.3f})")
    
    best_model = MODELS_DICT[best_model_name]
    best_model.fit(X, y)  # Full data fit
    
    save_model(best_model, 'best_model.pkl')
    save_model(scaler, 'scaler.pkl')
    save_model(selector, 'selector.pkl')
    save_model(encoders, 'encoder.pkl')
    save_metrics(metrics_df.to_dict())
    
    print("Complete!")
    print(metrics_df.round(3))
    return metrics_df, best_model_name

if __name__ == '__main__':
    train_models()
