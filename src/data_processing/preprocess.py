"""
Data Preprocessing & Feature Engineering Module
Prepares raw insurance data, validates schemas, engineers analytical features,
and constructs reproducible Scikit-Learn preprocessing transformers.
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERICAL_COLS = ['age', 'bmi', 'children']
CATEGORICAL_COLS = ['sex', 'smoker', 'region']
TARGET_COL = 'charges'

def assign_bmi_category(bmi: float) -> str:
    """Categorize BMI according to WHO standard classifications."""
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25.0:
        return 'Normal'
    elif bmi < 30.0:
        return 'Overweight'
    else:
        return 'Obese'

def assign_age_group(age: int) -> str:
    """Group chronological age into demographic segments."""
    if age < 30:
        return '18-29'
    elif age < 40:
        return '30-39'
    elif age < 50:
        return '40-49'
    else:
        return '50+'

def calculate_heuristic_risk_score(age: float, bmi: float, smoker: str, children: int = 0) -> float:
    """
    Calculate a normalized composite risk score (0 - 100) based on actuarial indicators:
    - Smoking status (primary contributor: +45 pts)
    - BMI classification (obesity threshold >= 30: up to +30 pts)
    - Age progression (scale 18 to 65: up to +20 pts)
    - Dependents factor (minor contributor: up to +5 pts)
    """
    score = 5.0  # baseline health risk

    # Smoking factor
    if str(smoker).lower().strip() == 'yes':
        score += 48.0

    # BMI factor
    if bmi >= 35.0:
        score += 27.0
    elif bmi >= 30.0:
        score += 20.0
    elif bmi >= 25.0:
        score += 8.0
    elif bmi < 18.5:
        score += 5.0

    # Compound interaction: obese + smoker
    if str(smoker).lower().strip() == 'yes' and bmi >= 30.0:
        score += 10.0

    # Age factor (18 to 65 mapped to 0 to 18)
    age_pts = max(0.0, min(18.0, (age - 18) * (18.0 / 47.0)))
    score += age_pts

    # Children factor (0 to 5 mapped to 0 to 4)
    child_pts = max(0.0, min(4.0, children * 0.8))
    score += child_pts

    return float(np.clip(round(score, 1), 0.0, 100.0))

def assign_risk_category(risk_score: float) -> str:
    """
    Documented Risk Threshold Logic:
    - LOW: Score < 35.0 (Younger/middle age, normal BMI or slight overweight, non-smoker)
    - MEDIUM: Score 35.0 to < 70.0 (Older individuals, elevated BMI, or intermediate profiles)
    - HIGH: Score >= 70.0 (Smokers, clinical obesity BMI >= 30, compound risk flags)
    """
    if risk_score < 35.0:
        return 'LOW'
    elif risk_score < 70.0:
        return 'MEDIUM'
    else:
        return 'HIGH'

def _vectorized_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized risk score computation — functionally equivalent to
    calling calculate_heuristic_risk_score row-by-row but significantly faster.
    """
    score = pd.Series(5.0, index=df.index)

    is_smoker = df['smoker'].str.lower().str.strip() == 'yes'
    score += is_smoker * 48.0

    bmi = df['bmi']
    score += (bmi >= 35.0) * 27.0
    score += ((bmi >= 30.0) & (bmi < 35.0)) * 20.0
    score += ((bmi >= 25.0) & (bmi < 30.0)) * 8.0
    score += (bmi < 18.5) * 5.0

    # Compound interaction: obese + smoker
    score += (is_smoker & (bmi >= 30.0)) * 10.0

    # Age factor (18 to 65 mapped to 0 to 18)
    age_pts = ((df['age'] - 18) * (18.0 / 47.0)).clip(lower=0.0, upper=18.0)
    score += age_pts

    # Children factor (0 to 5 mapped to 0 to 4)
    child_pts = (df['children'] * 0.8).clip(lower=0.0, upper=4.0)
    score += child_pts

    return score.clip(lower=0.0, upper=100.0).round(1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich dataframe with analytical categories and risk assessments."""
    df_copy = df.copy()

    # Derived categorical groupings
    df_copy['bmi_category'] = df_copy['bmi'].apply(assign_bmi_category)
    df_copy['age_group'] = df_copy['age'].apply(assign_age_group)

    # Risk scoring (vectorized for performance)
    df_copy['risk_score'] = _vectorized_risk_score(df_copy)
    df_copy['risk_category'] = df_copy['risk_score'].apply(assign_risk_category)

    return df_copy

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Create a robust Scikit-Learn ColumnTransformer for model consumption.
    Scales numerical features and one-hot encodes categorical dimensions.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ],
        remainder='drop'
    )
    return preprocessor

def run_data_pipeline(raw_path: str = None, processed_path: str = None) -> pd.DataFrame:
    """Execute end-to-end data processing and save processed artifact."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if raw_path is None:
        raw_path = os.path.join(base_dir, 'data', 'raw', 'insurance_dataset.csv')
    if processed_path is None:
        processed_path = os.path.join(base_dir, 'data', 'processed', 'insurance_processed.csv')

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}. Run generate_dataset.py first.")

    df_raw = pd.read_csv(raw_path)
    df_processed = engineer_features(df_raw)

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_processed.to_csv(processed_path, index=False)
    print(f"[Data Preprocessor] Processed dataset saved with {len(df_processed)} records at: {processed_path}")
    return df_processed

if __name__ == '__main__':
    run_data_pipeline()
