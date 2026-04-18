import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
import os
import sys
sys.path.insert(0, 'src')
from preprocessing import preprocess_data
from config import DATA_PROCESSED, FIGURES_DIR, DATA_RAW, TOP_FEATURES, MODELS_DIR
from utils import load_data

def detect_outliers(df, col):
    """Detect outliers using IQR."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (df[col] < lower) | (df[col] > upper)

st.set_page_config(layout="wide", page_title="Preprocessing", initial_sidebar_state="expanded")
st.markdown("""
<div class='metric-card'>
<h1 style='text-align: center; color: #4a5568;'>🔧 Data Preprocessing & Analysis</h1>
<p style='text-align: center; color: #718096;'>Outliers, missing values, feature engineering</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---", unsafe_allow_html=True)

# Raw data analysis
@st.cache_data(ttl=3600)
def analyze_raw():
    df = load_data(DATA_RAW)
    dups = df.duplicated().sum()
    missing = df.isnull().sum()
    num_cols = df.select_dtypes(np.number).columns.drop('mental_health_risk', errors='ignore')
    outliers = {}
    for col in num_cols:
        outliers[col] = detect_outliers(df, col).sum()
    outlier_df = pd.DataFrame(list(outliers.items()), columns=['Feature', 'Count'])
    skewness = df[num_cols].skew()
    target_dist = df['mental_health_risk'].value_counts(normalize=True)
    return df, dups, missing, outlier_df, num_cols, skewness, target_dist

df_raw, dups, missing, outlier_df, num_cols, raw_skew, raw_target = analyze_raw()

# Full-width metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Raw Rows", df_raw.shape[0])
col2.metric("Duplicates", dups)
col3.metric("Missing", missing.sum())
col4.metric("Outliers", outlier_df['Count'].sum() if not outlier_df.empty else 0)

# Raw data preview - responsive
st.subheader("📋 Raw Data Preview")
st.dataframe(df_raw.head(10), width="stretch")

# Issues summary
col_a, col_b = st.columns(2)
if dups > 0:
    with col_a:
        st.info(f"🔍 **{dups} duplicates found**")
        st.dataframe(df_raw[df_raw.duplicated(keep=False)].head(), use_container_width=True)

if missing.sum() > 0:
    with col_b:
        missing_pct = (missing / len(df_raw) * 100).round(2)
        missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
        st.dataframe(missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False), use_container_width=True)

# Visualizations
tab1, tab2, tab3 = st.tabs(["🎨 Target Distribution", "📊 Outliers", "🔗 Correlation & Skew"])

with tab1:
    fig_target = px.pie(values=raw_target.values, names=raw_target.index, 
                       title='Mental Health Risk Distribution', hole=0.4)
    st.plotly_chart(fig_target, use_container_width=True)

with tab2:
    if not outlier_df.empty:
        fig_out = px.bar(outlier_df.nlargest(10, 'Count'), x='Feature', y='Count', 
                        title='Top 10 Outliers by Feature', color='Count')
        st.plotly_chart(fig_out, use_container_width=True)

with tab3:
    col_corr, col_skew = st.columns(2)
    with col_corr:
        if len(num_cols) > 0:
            corr_raw = df_raw[num_cols].corr()
            fig_corr = px.imshow(corr_raw, title='Raw Numerical Correlation', 
                               color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with col_skew:
        st.subheader("Raw Skewness")
        skew_df = raw_skew.sort_values(ascending=False).round(2).to_frame('Skewness')
        st.dataframe(skew_df, use_container_width=True)

# CLI instruction
st.info("🚀 **Run full pipeline**: `python src/preprocessing.py`")
st.success("**Ready for modeling** after preprocessing!")

# Processed data analysis (if exists)
st.subheader("✅ Processed Data (Post-Pipeline)")
@st.cache_data(ttl=3600)
def analyze_processed():
    if os.path.exists(DATA_PROCESSED):
        df = pd.read_csv(DATA_PROCESSED)
        try:
            selector = joblib.load(os.path.join(MODELS_DIR, 'selector.pkl'))
            feature_scores = pd.Series(selector.scores_, index=selector.get_feature_names_out()).sort_values(ascending=False)
        except:
            feature_scores = pd.Series(dtype=float)
        num_cols_proc = df.select_dtypes(np.number).columns.drop('mental_health_risk', errors='ignore')
        skew_proc = df[num_cols_proc].skew()
        target_proc = df['mental_health_risk'].value_counts(normalize=True)
        return df, feature_scores, skew_proc, target_proc
    return None, None, None, None

df_proc, feat_scores, proc_skew, target_proc = analyze_processed()

if df_proc is not None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Processed Rows", df_proc.shape[0])
    col2.metric("Features", df_proc.shape[1]-1)
    col3.metric("Top Feature Score", feat_scores.iloc[0] if len(feat_scores)>0 else 0)
    col4.metric("Skew Reduced", "Yes" if abs(proc_skew).max() < 1.0 else "Partial")

    # Before/After comparison
    comp_data = {
        'Raw': [raw_skew.abs().max(), raw_target.max()],
        'Processed': [proc_skew.abs().max(), target_proc.max()]
    }
    comp_df = pd.DataFrame(comp_data, index=['Max Skew', 'Target Imbalance'])
    st.dataframe(comp_df.T.round(3), use_container_width=True)

    # Feature importance
    if len(feat_scores) > 0:
        fig_scores = px.bar(feat_scores.head(15), title='Top 15 Feature Scores (SelectKBest)')
        st.plotly_chart(fig_scores, use_container_width=True)
    
    # Processed preview
    st.subheader("Processed Data Preview")
    st.dataframe(df_proc.head(), use_container_width=True)

    st.success("✅ **Pipeline complete** - Ready for modeling!")
else:
    st.warning("🔄 **Run preprocessing first** to see processed analysis")

st.markdown("---")
st.caption("**Responsive full-width** - dups/missing/outliers → encode → SelectKBest(15) → scale")
