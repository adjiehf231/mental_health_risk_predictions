import streamlit as st
import pandas as pd
import sys
import joblib
import os
sys.path.insert(0, 'src')
import plotly.graph_objects as go
from predict import get_prediction

models = ['C4.5 (DT)', 'Random Forest', 'Naive Bayes', 'KNN', 'SVM']

st.header("🔮 Prediksi Risiko Kesehatan Mental")

model_choice = st.selectbox("Model", models, index=1)

# Accuracy info
acc = 0.0
star = ""
try:
    metrics_df = pd.read_csv('reports/model_comparison.csv', index_col=0)
    acc = metrics_df.loc[model_choice, 'Accuracy']
    best_model = metrics_df['Accuracy'].idxmax()
    if model_choice == best_model:
        star = " ⭐⭐⭐ Tertinggi!"
    st.metric("Accuracy", f"{acc:.3f}{star}")
except:
    st.metric("Accuracy", "0.85* (mock - train dulu)")

# Load selected features
try:
    selected_features = joblib.load('models/selected_features.pkl')
except:
    st.error("selected_features.pkl not found. Run preprocessing first.")
    st.stop()

st.subheader("Data Pasien (15 Selected Features)")
inputs = {}

cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status']
cat_options = {
    'gender': ['Male', 'Female'],
    'marital_status': ['Single', 'Married'],
    'education_level': ['High School', 'Bachelor', 'Master', 'PhD'],
    'employment_status': ['Employed', 'Unemployed']
}

num_defaults = {
    'age': (18, 60, 30),
    'sleep_hours': (3.0, 10.0, 7.0),
    'physical_activity_hours_per_week': (0, 20, 5),
    'screen_time_hours_per_day': (0, 24, 6),
    'social_support_score': (0, 10, 5),
    'work_stress_level': (0, 10, 5),
    'job_satisfaction_score': (0, 10, 7),
    'financial_stress_level': (0, 10, 4),
    'anxiety_score': (0, 10, 5),
    'depression_score': (0, 10, 5),
    'panic_attack_history': (0, 1, 0),
    'family_history_mental_illness': (0, 1, 0),
    'substance_use': (0, 1, 0)
}

for feat in selected_features:
    feat_label = feat.replace('_', ' ').title()
    
    if feat in cat_cols:
        options = cat_options.get(feat, ['Low', 'Medium', 'High'])
        inputs[feat] = st.selectbox(feat_label, options)
    else:
        if feat in num_defaults:
            min_val, max_val, default = num_defaults[feat]
            if isinstance(default, int):
                inputs[feat] = st.slider(feat_label, min_val, max_val, default)
            else:
                inputs[feat] = st.slider(feat_label, min_val, max_val, default, 0.1)
        else:
            inputs[feat] = st.slider(feat_label, 0, 10, 5)

if st.button("Prediksi", type="primary"):
    try:
        model_name = model_choice.lower().replace(' ', '_').replace('(', '').replace(')', '') + '.pkl'
        pred, proba, risk_label = get_prediction(inputs, model_name=model_name)
        
        col1, col2 = st.columns([3,1])
        col1.success(f"**Risiko: {risk_label}**")
        col2.info(f"Model: {model_choice} | Acc: {acc:.3f}{star}")
        
        fig = go.Figure([go.Bar(x=['Rendah', 'Sedang', 'Tinggi'], y=proba, marker_color=['green','orange','red'], text=[f'{p:.1%}' for p in proba], textposition='auto')])
        fig.update_layout(title="Probabilitas Risiko")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Model belum dilatih. Jalankan train_model. Error: {e}")
        st.info("Prediksi demo: Moderate Risk (1) | Acc 0.92")

st.caption("⭐ Model dengan Accuracy tertinggi | Train untuk hasil real.")
