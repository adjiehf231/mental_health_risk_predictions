import streamlit as st
import pandas as pd
import plotly.express as px
import sys
sys.path.insert(0, 'src')
import os

st.set_page_config(layout="wide", page_title="Modeling", initial_sidebar_state="expanded")
st.markdown("""
# 🤖 Model Comparison (K-Fold CV)
**Full-width responsive model analytics**
""")
st.markdown("---")

# CLI instruction
st.info("🚀 **Train via CLI**: `python src/train_model.py` (Cloud-safe)")
if os.path.exists('reports/model_comparison.csv'):
    st.success("✅ **Models trained** - Metrics live!")

# Load metrics
@st.cache_data(ttl=30)
def load_metrics():
    path = 'reports/model_comparison.csv'
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0).round(4)
        return df
    return None

df = load_metrics()

models = ['C4.5 (DT)', 'Random Forest', 'Naive Bayes', 'KNN', 'SVM']

if df is not None and not df.empty:
    st.success("✅ **Live training results**!")
    
    # Full metrics table
    st.subheader("📊 Complete Metrics Table")
    st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
    
    # Accuracy ranking
    st.subheader("🏆 Top Models by Accuracy")
    rank_df = df.sort_values('Accuracy', ascending=False)[['Accuracy', 'F1-Score', 'ROC-AUC']]
    st.dataframe(rank_df.style.background_gradient(cmap='viridis'), use_container_width=True)
    
    # Responsive charts
    col1, col2 = st.columns(2)
    with col1:
        # Bar chart
        fig = px.bar(df[['Accuracy', 'Precision', 'Recall', 'F1-Score']], 
                     barmode='group', 
                     title='Core Metrics Comparison',
                     color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        fig.update_layout(xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROC-AUC
        fig_auc = px.bar(df[['ROC-AUC']], x=df.index, y='ROC-AUC', 
                        title='ROC-AUC Scores',
                        color='ROC-AUC', color_continuous_scale='viridis')
        fig_auc.update_layout(height=450)
        st.plotly_chart(fig_auc, use_container_width=True)
    
    # Radar chart for top model
    top_model = df['Accuracy'].idxmax()
    top_metrics = df.loc[top_model]
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    fig_radar = px.line(pd.DataFrame({'Metric': categories, 'Score': top_metrics}), 
                       x='Metric', y='Score', line_shape='spline', title=f"Top Model: {top_model}")
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.balloons()
    st.markdown(f"**🎖️ #1 Model: {top_model}** - Acc: {top_metrics['Accuracy']:.3f}")
    
else:
    st.warning("🔄 **Run training first** for live metrics!")
    demo = pd.DataFrame({
        'Accuracy': [0.85, 0.92, 0.78, 0.87, 0.91],
        'Precision': [0.84, 0.91, 0.77, 0.86, 0.90],
        'Recall': [0.83, 0.90, 0.76, 0.85, 0.89],
        'F1-Score': [0.83, 0.91, 0.76, 0.85, 0.89],
        'ROC-AUC': [0.88, 0.95, 0.82, 0.89, 0.93]
    }, index=models)
    st.dataframe(demo.style.background_gradient(cmap='viridis'), use_container_width=True)
    st.caption("💡 Demo - Run training for **real 5-fold CV results**")

st.markdown("---")
st.caption("**DT(max_depth=10), RF(100 trees), GaussianNB, KNN(k=10), SVM(RBF)** | **5-Fold Macro Avg** | **Live updates**")

