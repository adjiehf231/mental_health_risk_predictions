import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw', 'mental_health_risk_dataset.csv')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', 'clean_data.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
METRICS_FILE = os.path.join(REPORTS_DIR, 'metrics.json')

# ML Config
RANDOM_STATE = 42
TEST_SIZE = 0.2
KFOLD = 5
TOP_FEATURES = 15

# Models
MODELS = {
    'C4.5 (DT)': 'DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10)',
    'Random Forest': 'RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)',
    'Naive Bayes': 'GaussianNB()',
    'KNN': 'KNeighborsClassifier(n_neighbors=5)',
    'SVM': 'SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)'
}
