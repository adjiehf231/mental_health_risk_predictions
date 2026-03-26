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

st.header("🔧 Data Preprocessing & Analysis")

# Raw data analysis
@st.cache_data
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

col1, col2 = st.columns(2)
col1.metric("Raw Rows", df_raw.shape[0])
col1.metric("Duplicates", dups)
col2.metric("Missing cells", missing.sum())
col2.metric("Outliers", outlier_df['Count'].sum() if not outlier_df.empty else 0)

# Raw data preview
st.subheader("📋 Raw Data Preview")
st.dataframe(df_raw.head(10))

if dups > 0:
    st.info(f"Found {dups} duplicates")
    st.dataframe(df_raw[df_raw.duplicated(keep=False)].head())

st.subheader("🎨 Raw Target Distribution")
fig_target = px.pie(values=raw_target.values, names=raw_target.index, title='Mental Health Risk Distribution')
st.plotly_chart(fig_target, use_container_width=True)

if missing.sum() > 0:
    missing_pct = (missing / len(df_raw) * 100).round(2)
    missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
    st.dataframe(missing_df[missing_df['Count'] > 0])

if not outlier_df.empty:
    fig_out = px.bar(outlier_df, x='Feature', y='Count', title='Outliers per Feature')
    st.plotly_chart(fig_out, use_container_width=True)

# Raw correlation
if len(num_cols) > 0:
    corr_raw = df_raw[num_cols].corr()
    fig_corr_raw = px.imshow(corr_raw, title='Raw Numerical Correlation', color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr_raw, use_container_width=True)

st.subheader("📊 Raw Skewness")
st.dataframe(raw_skew.sort_values(ascending=False).round(2))

# Pipeline button
if st.button("🚀 Run Full Preprocessing Pipeline", type="primary"):
    with st.spinner("Cleaning data..."):
        df_proc, selected_features, encoders, scaler, selector = preprocess_data()
    st.success("✅ Pipeline complete!")
    st.balloons()
    st.rerun()

# Processed comparison
st.subheader("✅ Fixed Data (After Pipeline)")
@st.cache_data
def analyze_processed():
    if os.path.exists(DATA_PROCESSED):
        df = pd.read_csv(DATA_PROCESSED)
        try:
            selector = joblib.load(os.path.join(MODELS_DIR, 'selector.pkl'))
            scores = selector.scores_
            feature_scores = pd.Series(scores, index=selected_features).sort_values(ascending=False)
        except:
            feature_scores = pd.Series(dtype=float)
        num_cols_proc = df.select_dtypes(np.number).columns.drop('mental_health_risk', errors='ignore')
        skew_proc = df[num_cols_proc].skew()
        target_proc = df['mental_health_risk'].value_counts(normalize=True)
        corr_proc = df[num_cols_proc].corr()
        return df, feature_scores, skew_proc, target_proc, corr_proc, num_cols_proc
    return None, None, None, None, None, None

df_proc, feat_scores, proc_skew, proc_target, corr_proc, num_cols_proc = analyze_processed()

if df_proc is not None:
    col1, col2 = st.columns(2)
    col1.metric("Fixed Rows", df_proc.shape[0])
    col2.metric("Selected Features", len(feat_scores))

    # Before/After comparison
    st.subheader("🔄 Before vs After Comparison")
    comp_df = pd.DataFrame({
        'Raw Skew Max': [raw_skew.abs().max()],
        'Proc Skew Max': [proc_skew.abs().max()],
        'Raw Target Balance': [raw_target.max()],
        'Proc Target Balance': [proc_target.max()]
    })
    st.dataframe(comp_df.round(3))

    # Feature scores
    if len(feat_scores) > 0:
        fig_scores = px.bar(x=feat_scores.index, y=feat_scores.values, title=f'Top {TOP_FEATURES} Feature Scores (SelectKBest)')
        st.plotly_chart(fig_scores, use_container_width=True)

    st.subheader("Fixed Data Preview")
    st.dataframe(df_proc.head(10))

    if len(num_cols_proc) > 0:
        fig_corr_proc = px.imshow(corr_proc, title='Processed Correlation', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr_proc, use_container_width=True)

        fig_target_proc = px.pie(values=proc_target.values, names=proc_target.index, title='Fixed Target Distribution')
        st.plotly_chart(fig_target_proc, use_container_width=True)

    st.subheader("Fixed Skewness")
    st.dataframe(proc_skew.sort_values(ascending=False).round(2))
    
    st.success("✅ Data ready for modeling! Reduced outliers, scaled features, top 15 selected.")
else:
    st.warning("🔄 Run pipeline first to see fixed data analysis.")

st.markdown("---")
st.caption("Pipeline: dups/missing/outliers → encode → SelectKBest(top15) → StandardScaler")
