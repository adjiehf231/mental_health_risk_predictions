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

# Fallback config
try:
    from .config import BASE_DIR, DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, METRICS_FILE, RANDOM_STATE, KFOLD
except:
    BASE_DIR = '.'
    DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw', 'mental_health_risk_dataset.csv')
    DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', 'clean_data.csv')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
    FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
    METRICS_FILE = os.path.join(REPORTS_DIR, 'metrics.json')
    RANDOM_STATE = 42
    KFOLD = 5

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

@st.cache_data
def load_data(path=DATA_RAW):
    return pd.read_csv(path)

def save_model(model, name):
    joblib.dump(model, os.path.join(MODELS_DIR, name))

def load_model(name):
    return joblib.load(os.path.join(MODELS_DIR, name))

def save_metrics(metrics):
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)

def plot_distribution(df, col):
    fig = px.histogram(df, x=col, marginal="violin", title=f'{col.replace("_", " ").title()} Distribution', nbins=30, opacity=0.8, color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(height=350, showlegend=False)
    return fig

def plot_correlation(df_num):
    plt.figure(figsize=(12,10))
    sns.heatmap(df_num.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.close()
    return None  # Use st.plotly_chart equivalent if needed

def plot_roc_multi(y_true, y_score):
    n_classes = 3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_onehot = label_binarize(y_true, classes=[0,1,2])
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fig = go.Figure()
    colors = ['green', 'orange', 'red']
    for i in range(n_classes):
        fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], name=f'Class {i} (AUC={roc_auc[i]:.2f})', line=dict(color=colors[i])))
    fig.add_shape(type="line", line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(title='Multi-Class ROC Curves', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    return fig

def plot_risk_psych_violins(df):
    psych_vars = ['anxiety_score', 'depression_score', 'stress_level', 'sleep_hours', 
                  'mood_swings_frequency', 'concentration_difficulty_level']
    psych_vars = [col for col in psych_vars if col in df.columns]
    
    fig = px.violin(df, x='mental_health_risk', y=psych_vars, box=True, points="outliers",
                    title="Psychological Variables by Risk Level",
                    color='mental_health_risk',
                    color_discrete_map={0:'green',1:'orange',2:'red'})
    fig.update_layout(height=500)
    return fig

def plot_psych_corr(df):
    psych_vars = ['anxiety_score', 'depression_score', 'stress_level', 'sleep_hours', 
                  'mood_swings_frequency', 'concentration_difficulty_level', 'mental_health_risk']
    psych_vars = [col for col in psych_vars if col in df.columns]
    if len(psych_vars) < 2:
        return None
    psych_df = df[psych_vars].corr()
    
    plt.figure(figsize=(10,8))
    sns.heatmap(psych_df, annot=True, cmap='RdBu_r', center=0, fmt='.2f', square=True)
    plt.title('Psychological Features + Risk Correlation')
    plt.tight_layout()
    plt.close()
    return None

def plot_risk_scatter_trends(df):
    psych_vars = ['anxiety_score', 'depression_score', 'age']
    fig = make_subplots(rows=1, cols=3, subplot_titles=psych_vars)
    
    jitter = 0.1
    colors = ['green', 'orange', 'red']
    for i, var in enumerate(psych_vars, 1):
        if var in df.columns:
            x_jitter = df['mental_health_risk'] + np.random.uniform(-jitter, jitter, len(df))
            fig.add_trace(go.Scatter(x=x_jitter, y=df[var], mode='markers',
                                   marker=dict(color=[colors[int(r)] for r in df['mental_health_risk']]), showlegend=False), row=1, col=i)
            z = np.polyfit(df['mental_health_risk'], df[var], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(x=df['mental_health_risk'], y=p(df['mental_health_risk']), 
                                   mode='lines', line=dict(dash='dash', color='black'), name=f'{var} trend'), row=1, col=i)
    
    fig.update_layout(height=400, title='Risk vs Key Features (jitter + trendline)')
    return fig

def plot_age_depression_trend(df):
    fig = px.scatter(df, x='age', y='depression_score', color='mental_health_risk',
                     color_discrete_map={0:'green',1:'orange',2:'red'}, trendline='ols',
                     title='Age vs Depression Score by Risk Level')
    fig.update_layout(height=400)
    return fig

def plot_age_risk_violin(df):
    fig = px.violin(df, x='mental_health_risk', y='age', box=True, 
                    title='Age Distribution by Risk Level',
                    color='mental_health_risk', 
                    color_discrete_map={0:'green',1:'orange',2:'red'})
    fig.update_layout(height=400)
    return fig

def plot_employment_trends(df):
    fig_risk = px.histogram(df, x='employment_status', color='mental_health_risk',
                            barmode='group', title='Risk Level by Employment Status',
                            color_discrete_map={0:'green',1:'orange',2:'red'})
    fig_risk.update_layout(height=400)
    
    fig_depression = px.box(df, x='employment_status', y='depression_score',
                            title='Depression Score by Employment Status')
    fig_depression.update_layout(height=400)
    
    return fig_risk, fig_depression

# Deployment safe - no file writing
print("Utils clean - deployment ready!")
