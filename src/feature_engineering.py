# Feature engineering module - advanced features if needed
# Currently minimal as basic preproc handles
# Can add PCA, interaction terms later

import pandas as pd
from sklearn.decomposition import PCA

def create_features(df):
    \"\"\"Create new features.\"\"\"
    df['stress_composite'] = (df['work_stress_level'] + df['academic_pressure_level'] + df['financial_stress_level']) / 3
    df['sleep_activity_ratio'] = df['sleep_hours'] / (df['physical_activity_hours_per_week'] + 1)
    return df

def apply_pca(df_features, n_components=10):
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(df_features)
    return pd.DataFrame(features_pca, columns=[f'PC{i+1}' for i in range(n_components)]), pca
