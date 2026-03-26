import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os

from .config import REPORTS_DIR, METRICS_FILE, BASE_DIR
from .utils import load_data

def load_metrics():
    """Load and display model metrics."""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
        df = pd.DataFrame(metrics).T
        return df
    return pd.DataFrame()

def plot_model_comparison(df):
    """Plot bar chart comparison."""
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df, x=df.index, y='F1-Score', title='F1 Score Comparison')
        st.plotly_chart(fig)
    with col2:
        fig = px.bar(df, x=df.index, y='Accuracy', title='Accuracy Comparison')
        st.plotly_chart(fig)

if __name__ == '__main__':
    st.title('Model Evaluation')
    df = load_metrics()
    if not df.empty:
        st.dataframe(df)
        plot_model_comparison(df)
    else:
        st.warning('No metrics found. Run train_model.py first.')
