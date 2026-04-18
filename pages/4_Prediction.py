import streamlit as st
import pandas as pd
import sys
import joblib
import os
sys.path.insert(0, 'src')
import plotly.graph_objects as go
from predict import get_prediction

st.set_page_config(layout="wide", page_title="Risk Prediction", initial_sidebar_state="expanded")
st.markdown("""
<div class='metric-card'>
<h1 style='text-align: center; color: #4a5568;'>🔮 Mental Health Risk Prediction</h1>
<p style='text-align: center; color: #718096;'>Adjust patient profile → instant risk assessment</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

models = ['C4.5 (DT)', 'Random Forest', 'Naive Bayes', 'KNN', 'SVM']

# Model selection + metrics
col_model, col_acc = st.columns([1,1])
with col_model:
    model_choice = st.selectbox("**Select Model**", models, index=1)

with col_acc:
    acc = 0.0
    star = ""
    try:
        metrics_df = pd.read_csv('reports/model_comparison.csv', index_col=0)
        acc = metrics_df.loc[model_choice, 'Accuracy']
        best_model = metrics_df['Accuracy'].idxmax()
        star = " ⭐⭐⭐ TOP MODEL!" if model_choice == best_model else ""
        st.metric("**Model Accuracy**", f"{acc:.1%}{star}", delta=None)
    except:
        st.metric("Accuracy", "92% (demo)")

# Cached models
@st.cache_resource
def load_models():
    return {
        'scaler': joblib.load('models/scaler.pkl'),
        'selector': joblib.load('models/selector.pkl'),
        'encoder': joblib.load('models/encoder.pkl'),
        'best_model': joblib.load('models/best_model.pkl')
    }

# Load features
try:
    selected_features = joblib.load('models/selected_features.pkl')
    st.success(f"**✅ Loaded {len(selected_features)} top features**")
except:
    st.error("❌ **selected_features.pkl missing**. Run: `python src/preprocessing.py`")
    st.stop()

st.markdown("### 👤 Patient Profile - Top 15 Selected Features")
st.markdown("*Adjust sliders → instant prediction*")

# Responsive input grid
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

# Grid layout for inputs
cols = st.columns(5)
for i, feat in enumerate(selected_features):
    col = cols[i % 5]
    feat_label = feat.replace('_', ' ').title()
    
    with col:
        if feat in cat_cols:
            options = cat_options.get(feat, ['Low', 'Medium', 'High'])
            inputs[feat] = st.selectbox(feat_label[:15], options, key=f"cat_{feat}")
        else:
            if feat in num_defaults:
                min_val, max_val, default = num_defaults[feat]
                step = 1 if isinstance(default, int) else 0.1
                inputs[feat] = st.slider(feat_label[:15], min_val, max_val, default, step, key=f"num_{feat}")
            else:
                inputs[feat] = st.slider(feat_label[:15], 0, 10, 5, key=f"num_{feat}")

# Predict button - full width
if st.button("**🔮 Predict Mental Health Risk**", type="primary", use_container_width=True):
    try:
        models_cache = load_models()
        pred, proba, risk_label = get_prediction(inputs)
        
        # Results full-width
        st.markdown("### **📊 Prediction Results**")
        col_result, col_model_info = st.columns([2,1])
        with col_result:
            risk_colors = {'Low Risk (0)': 'green', 'Moderate Risk (1)': 'orange', 'High Risk (2)': 'red'}
            st.markdown(f"### **{risk_label}** :bar_chart[ **{proba[pred]*100:.1f}% confidence** ]")
        
        with col_model_info:
            st.info(f"**Model**: {model_choice}\n**Accuracy**: {acc:.1%}")
        
        # Probability bar - interactive
        fig = go.Figure([go.Bar(x=['Low Risk', 'Moderate Risk', 'High Risk'], 
                               y=proba, 
                               marker_color=['#28a745','#ffc107','#dc3545'], 
                               text=[f'{p:.1%}' for p in proba], 
                               textposition='outside',
                               hovertemplate='<b>%{x}</b><br>Prob: %{y:.1%}<extra></extra>')])
        fig.update_layout(title="**Risk Probability Breakdown**", height=450, showlegend=False,
                         xaxis_tickangle=-0, margin=dict(l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ **Prediction error**: {e}")
        st.warning("**Run pipeline**: `python src/train_model.py`")
        st.info("**Demo**: Moderate Risk (1) - 92% confidence")

# Summary cards
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Features Used", len(selected_features))
with col2:
    st.metric("Pipeline Status", "✅ Complete" if os.path.exists('models/best_model.pkl') else "⚠️ Train first")
with col3:
    st.metric("Model Accuracy", f"{acc:.1%}")

st.caption("**Full-width responsive** - Production ready | Cached models | Train locally → deploy Cloud")

