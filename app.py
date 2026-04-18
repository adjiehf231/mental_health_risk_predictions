import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

st.set_page_config(
    page_title="Mental Health Risk Predictions",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding-top: 2rem;
}
.stApp {
    background: #f0f2f6;
}
h1, h2, h3 {
    color: #2d3748;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
button {
    border-radius: 25px;
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Mental Health Risk Predictions")
st.markdown("Portfolio Data Science - Streamlit Dashboard")

# Sidebar navigation
page_names_to_funcs = {
"📊 Dashboard & EDA": lambda: st.switch_page("pages/1_Dashboard.py"),
    "🔧 Data Preprocessing": lambda: st.switch_page("pages/2_Preprocessing.py"),
    "🤖 Machine Learning": lambda: st.switch_page("pages/3_Modeling.py"),
    "🔮 Predictions": lambda: st.switch_page("pages/4_Prediction.py")
}

selected_page = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]() 

# Footer
st.markdown("---")
st.markdown("Data source: [Kaggle Mental Health Dataset](https://www.kaggle.com/datasets/guriya79/mental-health-disorder)")
