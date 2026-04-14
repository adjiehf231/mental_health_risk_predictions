# Streamlit Cloud Deployment Fix - TODO
Status: 🚀 In Progress | Approved Plan

## Steps (Complete one-by-one, mark ✅ when done)

### 1. ✅ Fix requirements.txt
- Rewrite clean pinned versions  
- Test `pip install -r requirements.txt`

### 2. ✅ Optimize pages/1_Dashboard.py (Heavy CSV load)
### 3. ✅ Lazy-load models in pages/4_Prediction.py
### 4. ✅ Remove subprocess from pages/2_Preprocessing.py & 3_Modeling.py

### 5. ✅ Create .streamlit/config.toml (Optional)

### 6. ✅ Local test & Deploy
- `streamlit run app.py`
- Git commit/push  
- Redeploy Cloud → verify no spinner

**✅ Task complete!**

### 3. Lazy-load models in pages/4_Prediction.py
- Move joblib.load inside prediction button
- Add @st.cache_resource for models

### 4. Remove subprocess from pages/2_Preprocessing.py & 3_Modeling.py
- Replace with CLI instructions
- Add cache_data to analyses

### 5. Create .streamlit/config.toml (Optional)
- Server settings for Cloud

### 6. Local test & Deploy
- `streamlit run app.py`
- Git commit/push
- Redeploy Cloud → verify no spinner

**Next: Step 1**
