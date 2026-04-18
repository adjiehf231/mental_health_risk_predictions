# Mental Health Risk Prediction App 🧠

## Overview
Streamlit multipage app for predicting mental health risk levels (0=Low, 1=Moderate, 2=High) using Kaggle dataset (~25k rows, 26 cols: demographics, psych scores).

**Models**: DT (99.5% acc), RF, NB, KNN, SVM - KFold CV.
**Pipeline**: Raw → Preprocess (dedupe/outlier/encode/scale/SelectKBest=15) → Train → Predict.

## Structure
```
.
├── app.py (multipage nav)
├── pages/
│   ├── 1_Dashboard.py (full 25k EDA - plots/trends)
│   ├── 2_Preprocessing.py (analysis)
│   ├── 3_Modeling.py (metrics viz)
│   └── 4_Prediction.py (sliders → proba)
├── src/ (preprocessing.py, train_model.py, utils.py, config.py)
├── data/raw/mental_health_risk_dataset.csv
├── data/processed/
├── models/best_model.pkl (DT)
├── reports/model_comparison.csv
├── requirements.txt
├── run_pipeline.bat
└── TODO.md
```

## Quick Start
```bash
# 1. Pipeline (preprocess + train)
.\run_pipeline.bat

# 2. Run app
streamlit run app.py
```
**Live**: http://localhost:8501

## Features
- **Full EDA**: Risk dist, violins, correlations (25k raw data).
- **Preprocessing**: Outliers, skew reduction, feature selection.
- **Models**: 5 algos CV table/bar/radar.
- **Predict**: Interactive sliders (top 15 feats) → risk/proba chart.

## Deploy
1. GitHub repo.
2. Streamlit Cloud (connect repo, auto-deps).

## Screenshots
*(Add after run)*

## Dependencies
Pinned in `requirements.txt` (streamlit 1.36, sklearn 1.4, plotly 5.0).

**Best Model**: C4.5 DT - Acc 99.5%, F1 99.3%!

Made with ❤️ by BLACKBOXAI
