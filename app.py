import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

st.set_page_config(
    page_title="Mental Health Risk Predictions",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
