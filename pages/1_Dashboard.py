import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
sys.path.insert(0, 'src')

from src.utils_clean import load_data, plot_risk_psych_violins, plot_psych_corr, plot_risk_scatter_trends, plot_age_depression_trend, plot_age_risk_violin, plot_employment_trends
from config import DATA_RAW, FIGURES_DIR

st.set_page_config(layout="wide", page_title="Mental Health Dashboard", initial_sidebar_state="expanded")
st.markdown("""
<div class='metric-card'>
<h1 style='text-align: center; color: #4a5568;'>📊 Data Overview & EDA</h1>
<p style='text-align: center; color: #718096;'>Full dataset insights - interactive visualizations</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---", unsafe_allow_html=True)

# Load data - full dataset
@st.cache_data(ttl=3600)
def load_raw():
    df_full = load_data(DATA_RAW)
    return df_full

df = load_raw()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Low Risk", (df['mental_health_risk'] == 0).sum())
col4.metric("High Risk", (df['mental_health_risk'] == 2).sum())

# Dataset info
with st.expander("📋 Dataset Info"):
    st.markdown("""
    **Kaggle**: [Mental Health Dataset](https://www.kaggle.com/datasets/guriya79/mental-health-disorder)
    **25 features** (demographics, psych scores) → **Risk 0-2**
    """)

# Risk distribution pie
@st.cache_data
def pie_chart(df):
    fig = px.pie(df, names='mental_health_risk', hole=0.4, 
                 color_discrete_sequence=['green', 'orange', 'red'],
                 title='Risk Distribution')
    fig.update_layout(height=450, showlegend=True)
    return fig

st.plotly_chart(pie_chart(df), use_container_width=True)

# Key features histograms
key_features = ['age', 'sleep_hours', 'anxiety_score', 'depression_score', 'stress_level']
cols = st.columns(3)
for i, feat in enumerate(key_features):
    if feat in df.columns:
        with cols[i % 3]:
            fig = px.histogram(df[feat], nbins=30, title=feat.replace('_', ' ').title())
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

# Trends section
st.markdown("### 🔬 Key Trends")
tab1, tab2 = st.tabs(["Psych vs Risk", "Demographics"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig_violins = plot_risk_psych_violins(df)
        st.plotly_chart(fig_violins, use_container_width=True)
    with col2:
        fig_scatter = plot_risk_scatter_trends(df)
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        fig_age_dep = plot_age_depression_trend(df)
        st.plotly_chart(fig_age_dep, use_container_width=True)
    with col2:
        fig_age_risk = plot_age_risk_violin(df)
        st.plotly_chart(fig_age_risk, use_container_width=True)

# Employment trends
fig_emp_risk, fig_emp_dep = plot_employment_trends(df)
col1, col2 = st.columns(2)
st.plotly_chart(fig_emp_risk, use_container_width=True)
st.plotly_chart(fig_emp_dep, use_container_width=True)

# Correlation image
corr_img = os.path.join(FIGURES_DIR, 'correlation.png')
if os.path.exists(corr_img):
    st.image(corr_img, caption="Correlation Heatmap", use_container_width=True)
else:
    st.info("🔄 Run preprocessing for correlation heatmap")

st.markdown("---")
st.success("✅ Interactive EDA - Ready for modeling!")
st.caption("Next: 🔧 Preprocessing → 🤖 Models → 🔮 Predict")
