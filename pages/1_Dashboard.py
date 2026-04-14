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
# 📊 Data Overview & EDA
**Responsive full-width layout with interactive charts**
""")
st.markdown("---", unsafe_allow_html=True)

# Load data - sampled for Cloud performance
@st.cache_data(ttl=3600)
def load_raw():
    df_full = load_data(DATA_RAW)
    return df_full.sample(n=min(1000, len(df_full)), random_state=42)

df = load_raw()

# Full-width metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sample Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Risk 0 (Low)", (df['mental_health_risk'] == 0).sum())
col4.metric("Risk 2 (High)", (df['mental_health_risk'] == 2).sum())

# Data source expander
with st.expander("📋 Dataset Info", expanded=False):
    st.markdown("""
    **Kaggle Dataset**: [Mental Health Disorder Prediction](https://www.kaggle.com/datasets/guriya79/mental-health-disorder)
    
    **Features**: 25 (demographics, lifestyle, stress, psychological, medical history)
    **Target**: mental_health_risk (0=Low, 1=Moderate, 2=High)
    """)
    st.info("📋 25 features, demographics + psych scores. Target: risk levels 0-2")

# Cached target pie - full width
@st.cache_data(ttl=3600)
def pie_chart_cached(df):
    fig = px.pie(df, names='mental_health_risk', 
                 title='<b>Mental Health Risk Distribution (Sample)</b>',
                 color_discrete_sequence=['#2E8B57', '#FF8C00', '#DC143C'],
                 hole=0.4)
    fig.update_layout(font=dict(size=14), height=450, showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig

st.plotly_chart(pie_chart_cached(df), use_container_width=True)

# Key features histograms - responsive grid
@st.cache_data(ttl=3600)
def histogram_cached(df_sample, feature):
    fig = px.histogram(df_sample, x=feature,
                       color_discrete_sequence=px.colors.qualitative.Set3,
                       title=f'{feature.replace("_", " ").title()}',
                       marginal='violin', nbins=15, opacity=0.8,
                       labels={'value': feature.title()})
    fig.update_layout(font=dict(size=12), plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)', height=400,
                      title_font_size=16, showlegend=False, margin=dict(l=20, r=20))
    fig.update_traces(marker_line_color='white', marker_line_width=1)
    return fig

key_features = ['age', 'sleep_hours', 'screen_time_hours_per_day', 
                'anxiety_score', 'depression_score', 'stress_level']
df_sample = df.sample(n=min(500, len(df)), random_state=42)

cols = st.columns(3)
for i, feature in enumerate(key_features):
    if feature in df_sample.columns:
        with cols[i%3]:
            st.plotly_chart(histogram_cached(df_sample, feature), use_container_width=True)

# Static correlation
corr_img = os.path.join(FIGURES_DIR, 'correlation.png')
if os.path.exists(corr_img):
    st.image(corr_img, caption="Correlation Heatmap (Numerical Features)", use_container_width=True)
else:
    st.info("🔄 Generate via preprocessing for static fast load")

# Full-width trends section
st.markdown("### 🔬 Psychological & Demographic Trends")
tab1, tab2 = st.tabs(["Risk vs Psych", "Demographics"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Violin by Risk**")
        fig_violins = plot_risk_psych_violins(df)
        st.plotly_chart(fig_violins, use_container_width=True)
    
    with col2:
        psych_img = os.path.join(FIGURES_DIR, 'psych_corr.png')
        if os.path.exists(psych_img):
            st.image(psych_img, caption="Psych Correlation", use_container_width=True)
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
col_emp1, col_emp2 = st.columns(2)
col_emp1.plotly_chart(fig_emp_risk, use_container_width=True)
col_emp2.plotly_chart(fig_emp_dep, use_container_width=True)

st.markdown("---")
st.success("✅ **Full-width responsive EDA** - Fast Cloud loading with interactive charts!")
st.caption("Next: 🔧 Preprocessing → 🤖 Modeling → 🔮 Prediction")
