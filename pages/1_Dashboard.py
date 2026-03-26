import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
sys.path.insert(0, 'src')


from src.utils_clean import load_data, plot_risk_psych_violins, plot_psych_corr, plot_risk_scatter_trends, plot_age_depression_trend, plot_age_risk_violin, plot_employment_trends
import sys
sys.path.insert(0, 'src')


from config import DATA_RAW, FIGURES_DIR

st.header("📊 Data Overview & EDA")

# Load data
@st.cache_data
def load_raw():
    return load_data(DATA_RAW)

df = load_raw()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Risk 0 (Low)", (df['mental_health_risk'] == 0).sum())
col4.metric("Risk 2 (High)", (df['mental_health_risk'] == 2).sum())

# Data source
with st.expander("Data Source"):
    st.markdown("""
    **Kaggle Dataset**: [Mental Health Disorder Prediction](https://www.kaggle.com/datasets/guriya79/mental-health-disorder)
    
    **Features**: 25 (demographics, lifestyle, stress, psychological, medical history)
    **Target**: mental_health_risk (0=Low, 1=Moderate, 2=High)
    """)
    st.info("📋 Dataset: 25 features, demographics + psych scores. Target: risk (0=Low,1=Mod,2=High)")

# Target distribution

fig_pie = px.pie(df, names='mental_health_risk', 
                 title='<b>Mental Health Risk Distribution</b>',
                 color_discrete_sequence=['#2E8B57', '#FF8C00', '#DC143C'],
                 hole=0.4)
fig_pie.update_layout(
    font=dict(size=14),
    height=400,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=-0.2)
)
st.plotly_chart(fig_pie, use_container_width=True)


# Key features distribution
key_features = ['age', 'sleep_hours', 'screen_time_hours_per_day', 
                'anxiety_score', 'depression_score', 'stress_level']


for feature in key_features:
    fig = px.histogram(df, x=feature, 
                       color_discrete_sequence=px.colors.qualitative.Set3,
                       title=f'<b>{feature.replace("_", " ").title()}</b>',
                       marginal='violin',
                       nbins=25, opacity=0.8,
                       labels={'value': feature.title()})
    fig.update_layout(

        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=350,
        title_font_size=16,
        showlegend=False
    )
    fig.update_traces(marker_line_color='white', marker_line_width=1)
    st.plotly_chart(fig, use_container_width=True)


# Correlation heatmap
if os.path.exists(os.path.join(FIGURES_DIR, 'correlation.png')):
    st.image(os.path.join(FIGURES_DIR, 'correlation.png'), 
             caption="Correlation Heatmap (Numerical Features)")
else:
    st.info("Run preprocessing/train to generate plots")

# Enhanced Mental Health Trends
st.subheader("🔬 Mental Health Risk & Psychological Trends")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Violin Plots by Risk**")
    fig_violins = plot_risk_psych_violins(df)
    st.plotly_chart(fig_violins, use_container_width=True)

with col2:
    if os.path.exists(os.path.join(FIGURES_DIR, 'psych_corr.png')):
        st.image(os.path.join(FIGURES_DIR, 'psych_corr.png'), 
                 caption="Psychological Features Corr Heatmap", use_column_width=True)
    else:
        plot_psych_corr(df)
        st.image(os.path.join(FIGURES_DIR, 'psych_corr.png'), caption="Psych Corr")


fig_scatter = plot_risk_scatter_trends(df)
st.plotly_chart(fig_scatter, use_container_width=True)

# Demographic & Status Trends
st.subheader("👥 Demographic & Employment Trends")
fig_age_dep = plot_age_depression_trend(df)
st.plotly_chart(fig_age_dep, use_container_width=True)

fig_age_risk = plot_age_risk_violin(df)
st.plotly_chart(fig_age_risk, use_container_width=True)

fig_emp_risk, fig_emp_dep = plot_employment_trends(df)
col_emp1, col_emp2 = st.columns(2)
col_emp1.plotly_chart(fig_emp_risk, use_container_width=True)
col_emp2.plotly_chart(fig_emp_dep, use_container_width=True)

st.success("Full explorative EDA complete! Preprocessing → Modeling → Prediction")


