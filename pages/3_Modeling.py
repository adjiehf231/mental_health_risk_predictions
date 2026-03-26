import streamlit as st
import pandas as pd
import plotly.express as px
import sys
sys.path.insert(0, 'src')
import os

st.header("🤖 Model Comparison (K-Fold CV)")

# Train button
if st.button("🚀 Train All Models", type="primary"):
    with st.spinner('Training 5 models...'):
        import subprocess
        subprocess.run(['.venv\\Scripts\\python.exe', '-m', 'src.train_model'], shell=True)
    st.rerun()

# Load
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
    st.success("✅ Data training real loaded!")
    
    st.subheader("Metrics Lengkap")
    st.dataframe(df)
    
    st.subheader("🏆 Ranking Accuracy")
    rank_df = df.sort_values('Accuracy', ascending=False)[['Accuracy']]
    st.dataframe(rank_df)
    st.balloons()
    st.markdown(f"**#1 {rank_df.index[0]}: {rank_df['Accuracy'].iloc[0]:.3f}**")
    
    # Full metrics bar chart
    st.subheader("Diagram Metrics Lengkap")
    fig = px.bar(df[['Accuracy', 'Precision', 'Recall', 'F1-Score']], 
                 barmode='group', 
                 title='Accuracy, Precision, Recall, F1-Score per Algoritma',
                 color_discrete_sequence=['blue', 'green', 'orange', 'red'])
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC-AUC
    fig_auc = px.bar(df['ROC-AUC'], x=df.index, title='ROC-AUC', color='ROC-AUC')
    st.plotly_chart(fig_auc)
    
else:
    st.warning("Klik Train untuk data real!")
    demo = pd.DataFrame({
        'Accuracy': [0.85, 0.92, 0.78, 0.87, 0.91],
        'Precision': [0.84, 0.91, 0.77, 0.86, 0.90],
        'Recall': [0.83, 0.90, 0.76, 0.85, 0.89],
        'F1-Score': [0.83, 0.91, 0.76, 0.85, 0.89]
    }, index=models)
    st.dataframe(demo)
    st.caption("Demo")

st.caption("DT depth10, RF 100trees, NB, KNN5, SVM RBF | 5Fold Macro Avg | Auto-update!")

