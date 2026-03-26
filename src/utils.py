import pandas as pd
import numpy as np
import os
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
import streamlit as st

# To support both package import and direct import
try:
    from .config import BASE_DIR, DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, \
                        FIGURES_DIR, METRICS_FILE, RANDOM_STATE, KFOLD
except (ImportError, ValueError):
    BASE_DIR = '.'
    DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw', 'mental_health_risk_dataset.csv')
    DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', 'clean_data.csv')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
    FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
    METRICS_FILE = os.path.join(REPORTS_DIR, 'metrics.json')
    RANDOM_STATE = 42
    KFOLD = 10

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

@st.cache_data
def load_data(path=DATA_RAW):
    """Load raw data."""
    return pd.read_csv(path)

def save_model(model, name):
    joblib.dump(model, os.path.join(MODELS_DIR, name))

def load_model(name):
    return joblib.load(os.path.join(MODELS_DIR, name))

def save_metrics(metrics):
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)

def plot_distribution(df, col):
    fig = px.histogram(df, x=col, marginal="box", title=f'Distribution of {col}')
    return fig

def plot_correlation(df_num):
    plt.figure(figsize=(12,10))
    sns.heatmap(df_num.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation.png'))
    plt.close()

def plot_roc_multi(y_true, y_score, n_classes=3):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_binarize(y_true, classes=[0,1,2])[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fig = go.Figure()
    for i in range(n_classes):
        fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], name=f'Class {i} (AUC={roc_auc[i]:.2f})'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random'))

    fig.update_layout(title='ROC Curves', xaxis_title='FPR', yaxis_title='TPR')
    return fig



def plot_risk_psych_violins(df):
    """Violin plots: psych vars by risk level."""


    psych_vars = ['anxiety_score', 'depression_score', 'stress_level', 'sleep_hours', 
                  'mood_swings_frequency', 'concentration_difficulty_level']
    psych_vars = [col for col in psych_vars if col in df.columns]
    
    fig = px.violin(df, x='mental_health_risk', y=psych_vars, box=True, points="outliers",
                    title="Psychological Variables by Risk Level (0-Low,1-Mod,2-High)",
                    labels={'mental_health_risk': 'Risk Level'},
                    color='mental_health_risk',
                    color_discrete_map={0:'green',1:'orange',2:'red'})
    fig.update_layout(height=500)
    fig.write_image(os.path.join(FIGURES_DIR, 'risk_psych_violins.png'))
    return fig



def plot_psych_corr(df):
    """Psych + target corr heatmap."""


    psych_vars = ['anxiety_score', 'depression_score', 'stress_level', 'sleep_hours', 
                  'mood_swings_frequency', 'concentration_difficulty_level', 'mental_health_risk']
    psych_vars = [col for col in psych_vars if col in df.columns]
    if len(psych_vars) < 2:
        return None
    psych_df = df[psych_vars].corr()
    
    plt.figure(figsize=(10,8))
    sns.heatmap(psych_df, annot=True, cmap='RdBu_r', center=0, fmt='.2f', square=True)
    plt.title('Mental Health Psychological Features Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'psych_corr.png'), dpi=300, bbox_inches='tight')
    plt.close()



def plot_risk_scatter_trends(df):
    """Trend scatters risk vs psych with jitter/trendline."""


    psych_vars = ['anxiety_score', 'depression_score']
    fig = make_subplots(rows=1, cols=2, subplot_titles=psych_vars)
    
    jitter = 0.1
    colors = ['green', 'orange', 'red']
    for i, var in enumerate(psych_vars, 1):
        if var in df.columns:
            x_jitter = df['mental_health_risk'] + np.random.uniform(-jitter, jitter, len(df))
            scatter = go.Scatter(x=x_jitter, y=df[var], mode='markers', 
                                marker=dict(color=[colors[int(r)] for r in df['mental_health_risk']]),
                                name=var, showlegend=False)
            fig.add_trace(scatter, row=1, col=i)
            # Trendline
            z = np.polyfit(df['mental_health_risk'], df[var], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(x=df['mental_health_risk'], y=p(df['mental_health_risk']), 
                                   mode='lines', name='trend', line=dict(dash='dash')), row=1, col=i)
    
    fig.update_layout(height=400, title='Risk Level vs Key Psych Scores (jittered + trend)')
    fig.write_image(os.path.join(FIGURES_DIR, 'risk_scatter_trends.png'))
    return fig

