# 🧠 **Mental Health Risk Prediction - Complete Data Science Solution**

## **1. Project Overview**
**End-to-End Data Science Portfolio Project**: Interactive Streamlit dashboard implementing full CRISP-DM process for **multi-class mental health risk classification** from Kaggle dataset.

### **Problem Statement**
Predict mental health risk levels (**Low=0, Moderate=1, High=2**) using 25 behavioral/demographic/psychological features.

### **Business Value**
- **Early Detection**: Identify high-risk individuals
- **Resource Allocation**: Prioritize therapy/intervention
- **Population Health**: Track trends by age/employment/status

## **2. Dataset - Detailed Analysis**
**Source**: [Kaggle Mental Health Disorder](https://www.kaggle.com/datasets/guriya79/mental-health-disorder)

| **Category** | **Features (15/25 shown)** | **Type** | **Description** |
|--------------|----------------------------|----------|-----------------|
| **Demographics** (5) | `age`, `gender`, `marital_status`, `education_level` | Num/Cat | Population characteristics |
| **Lifestyle** (6) | `sleep_hours`, `physical_activity_hours_per_week`, `screen_time_hours_per_day`, `social_support_score` | Num | Daily habits |
| **Stress** (5) | `work_stress_level`, `academic_pressure_level`, `financial_stress_level` | Num 1-10 | Stress sources |
| **Psychological** (6) | `anxiety_score`, `depression_score`, `stress_level`, `mood_swings_frequency`, `concentration_difficulty_level` | Num 1-10 | Mental health scores |
| **Medical** (3) | `panic_attack_history`, `family_history_mental_illness`, `previous_mental_health_diagnosis` | Binary | Medical history |

**Target**: `mental_health_risk` (0=Low, 1=Moderate, 2=High)

**EDA Insights** (from Dashboard):
```
Risk Distribution: Low(48%), Moderate(32%), High(20%)
Correlation Leaders: depression_score(r=0.72), anxiety_score(r=0.68), stress_level(r=0.65)
Age Effect: Mean age High-risk = 42 vs Low-risk = 32
Employment: Unemployed depression median = 7.8 (vs 4.2 employed)
```

## **3. Data Science Pipeline - Technical Details**

### **3.1 Preprocessing (src/preprocessing.py)**
```
Input: raw CSV (25k x 26)
1. Duplicate removal: -2.1% samples
2. Missing value imputation: mode(cat)/mean(num)
3. Outlier capping: 1st/99th percentile (IQR detection)
4. Categorical encoding: LabelEncoder (4 vars)
5. Feature selection: SelectKBest(f_classif, k=15)
6. Scaling: StandardScaler
Output: clean_data.csv (25k x 16: 15 features + target)
Artifacts: encoder/scaler/selector.pkl
```

### **3.2 Feature Engineering (src/feature_engineering.py)**
```
Composite features:
- stress_composite = (work_stress + academic_pressure + financial_stress) / 3
- sleep_activity_ratio = sleep_hours / (physical_activity + 1)
```

### **3.3 Model Training (src/train_model.py)**
```
Validation: 5-fold StratifiedKFold + 80/20 train/test split
Models & Hyperparameters:
1. DecisionTree: max_depth=10, random_state=42
2. RandomForest: n_estimators=100, random_state=42
3. GaussianNB: default
4. KNN: n_neighbors=5
5. SVM: kernel='rbf', probability=True

Metrics (macro-average):
- Accuracy, Precision, Recall, F1-Score, ROC-AUC (OvR)
Best model saved by F1-score
```

**Sample Results**:
```
Model | Acc | Prec | Recall | F1 | ROC
------|-----|------|--------|----|-----
RF | 0.89| 0.89 | 0.88 | **0.88** | 0.94
SVM| 0.87| 0.87 | 0.87 | 0.87 | 0.93
```

### **3.4 Prediction Pipeline**
```
raw_input → encoders → selector.transform() → scaler.transform() → best_model.predict_proba()
Output: class + [p_low, p_mod, p_high]
```

## **4. Streamlit Dashboard - Page Details**

### **Page 1: Interactive EDA (pages/1_Dashboard.py)**
**Modern Visualizations** (Plotly responsive):
```
1. Metrics KPI cards (rows=25k, cols=26, risk counts)
2. Modern Donut Pie Chart (risk distribution - Set3 colors, horizontal legend)
3. 6 Key Distribution Plots (age/stress modern violin marginals, transparent design)
4. Risk vs psych violin swarm (6 variables)
5. Psych correlation heatmap (11x11 matrix)
6. Risk jitter scatters + trendlines (anxiety/depression/age - 3 subplot)
7. Age-depression bivariate scatter (risk colored)
8. Age violin by risk category
9. Employment-risk distribution (stacked bar)
10. Employment-depression boxplot
11. Full correlation heatmap image
```

### **Page 2: Preprocessing (pages/2_Preprocessing.py)**
```
One-click pipeline + raw vs processed comparison
Selected top-15 features list
Statistics before/after
```

### **Page 3: Modeling Results (pages/3_Modeling.py)**
```
CV metrics table (sortable)
F1-score & ROC-AUC bar charts
Individual ROC curves gallery
Best model highlight
```

### **Page 4: Prediction (pages/4_Prediction.py)**
```
Interactive form (15 sliders/dropdowns)
Live predictions
Probability bars (Low/Mod/High %)
Input summary
Error handling
```

## **5. File Structure & Generated Artifacts**

```
mental_health_risk_predictions/
├── app.py (Streamlit multipage controller)
├── config.py (paths/hyperparameters)
├── requirements.txt (19 dependencies)
├── README.md (detailed documentation)

**Data (Generated)**
├── data/raw/mental_health_risk_dataset.csv (25k x 26 original)
├── data/processed/clean_data.csv (25k x 16 preprocessed)
├── data/processed/selected_features.pkl

**Models (Generated)**
├── models/best_model.pkl (~1MB - highest F1)
├── models/scaler.pkl (StandardScaler)
├── models/encoder.pkl (LabelEncoder)
├── models/selector.pkl (SelectKBest)
├── models/selected_features.pkl

**Reports (Generated)**
├── reports/metrics.json (raw model scores)
├── reports/model_comparison.csv (formatted table)
├── reports/figures/
    ├── correlation.png
    ├── risk_psych_violins.png
    ├── psych_corr.png
    ├── risk_scatter_trends.png
    ├── age_depression_trend.png
    ├── age_by_risk.png

**Source Code**
├── src/
│   ├── config.py (ML hyperparameters)
│   ├── utils_fixed.py (12 modern Plotly functions)
│   ├── preprocessing.py (full ETL)
│   ├── train_model.py (5-fold CV pipeline)
│   ├── predict.py (model serving)
│   ├── evaluate.py (metrics viz)
│   └── feature_engineering.py (composites)

**UI Pages**
└── pages/
    ├── 1_Dashboard.py (20+ interactive plots)
    ├── 2_Preprocessing.py
    ├── 3_Modeling.py
    └── 4_Prediction.py
```

## **6. Complete Setup & Execution Guide**

### **Step 1: Environment**
```bash
cd "d:/PROGRAM/DATA SCIENCE/mental_health_risk_predictions"
.venv/Scripts/activate
pip install -r requirements.txt
```

### **Step 2: Data Science Pipeline**
```bash
cd src
python train_model.py  # 2 minutes - generates everything
cd ..
```

### **Step 3: Interactive Dashboard**
```bash
streamlit run app.py
```
**URL**: `http://localhost:8501`

## **7. Expected Training Output**
```
✓ Loaded 25000 samples
✓ Removed 525 duplicates (2.1%)
✓ Handled 0 missing values
✓ Capped outliers in 18 numerical columns
✓ Encoded 4 categorical features
✓ Selected top 15 features: ['anxiety_score', 'depression_score', ...]
✓ Saved clean_data.csv
Training C4.5 (DT)... Done
Training Random Forest... Done
Training Naive Bayes... Done
Training KNN... Done  
Training SVM... Done
Training complete!
Best model: Random Forest (F1: 0.88)
✓ All models saved to models/
✓ Metrics → reports/
✓ 6 figures generated
```

## **8. Model Technical Specifications**
```
**Validation Strategy**:
- Train/Test: 80/20 stratified split (random_state=42)
- Cross Validation: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

**Hyperparameters**:
DecisionTreeClassifier(max_depth=10, random_state=42)
RandomForestClassifier(n_estimators=100, random_state=42)
GaussianNB()
KNeighborsClassifier(n_neighbors=5)
SVC(kernel='rbf', probability=True, random_state=42)

**Evaluation Metrics** (macro-average all classes):
accuracy_score | precision_score | recall_score | f1_score | roc_auc_score(ovr)
```

## **9. Production Deployment Guide**
```
**Streamlit Cloud** (Free):
1. Push to GitHub
2. streamlit.io → New app → Select repo/branch/app.py
3. Auto-deploys

**Docker**:
docker build -t mental-health-app .
docker run -p 8501:8501 mental-health-app
```

## **10. Screenshot Gallery**
```
Dashboard Metrics + Modern Distributions
[Insert Screenshot 1]

Psych Trends + Risk Scatters
[Insert Screenshot 2]

Demographic Analysis
[Insert Screenshot 3]

Model Results + Prediction
[Insert Screenshot 4]
```

## **👨‍💻 Credits**
**Adjie Hari Fajar**  
**Data Scientist | Data Analyst | Machine Learning Engineer**  
**Python Programmer | RStudio | IT Support | QA Engineer**

---

**Status**: 🎯 **Complete Data Science Solution**  
**Ready**: `streamlit run app.py`  
**Reproducible**: Full specs + seeds  
**Production**: Deploy-ready

