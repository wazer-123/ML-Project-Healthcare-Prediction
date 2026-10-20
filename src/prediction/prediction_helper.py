"""
Prediction Helper Module
Handles model loading, feature validation, inference generation,
risk assessment, and non-causal explanation generation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.data_processing.preprocess import (
    assign_bmi_category,
    assign_age_group,
    calculate_heuristic_risk_score,
    assign_risk_category,
    NUMERICAL_COLS,
    CATEGORICAL_COLS
)

class InsurancePredictor:
    """Encapsulates model artifacts and inference logic."""

    def __init__(self, model_dir: str | None = None):
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_dir = os.path.join(base_dir, 'models')

        self.model_path = os.path.join(model_dir, 'best_model.pkl')
        self.preprocessor_path = os.path.join(model_dir, 'preprocessing_pipeline.pkl')
        self.all_models_path = os.path.join(model_dir, 'all_models.pkl')

        self.model = None
        self.preprocessor = None
        self.all_models = {}
        self.is_loaded = False
        self.load_artifacts()

    def load_artifacts(self) -> bool:
        """Load trained model and preprocessing pipeline."""
        if os.path.exists(self.model_path) and os.path.exists(self.preprocessor_path):
            try:
                self.model = joblib.load(self.model_path)
                self.preprocessor = joblib.load(self.preprocessor_path)
                if os.path.exists(self.all_models_path):
                    self.all_models = joblib.load(self.all_models_path)
                self.is_loaded = True
                return True
            except Exception as e:
                print(f"[Error loading artifacts]: {e}")
                self.is_loaded = False
                return False
        return False

    def predict_single(
        self,
        age: int | float | str,
        sex: float | int | str,
        bmi: float | int | str,
        children: int | float | str,
        smoker: float | int | str,
        region: float | int | str,
        model_name: str | None = None
    ) -> Dict[str, Any]:
        """
        Generate prediction and actuarial risk profile for an individual profile.
        """
        if not self.is_loaded or self.preprocessor is None or self.model is None:
            if not self.load_artifacts():
                raise RuntimeError("Model or Preprocessor artifact not found. Please train models first.")
        if self.preprocessor is None or self.model is None:
            raise RuntimeError("Artifacts loaded but preprocessor or model is None. Re-train and export models.")

        # Robust input validation & sanitization (handling None, NaN, and invalid types)
        try:
            age_f = float(age)
            if np.isnan(age_f) or np.isinf(age_f):
                age_f = 35.0
        except (TypeError, ValueError):
            age_f = 35.0
        age_clean = max(18, min(100, round(age_f)))

        try:
            bmi_f = float(bmi)
            if np.isnan(bmi_f) or np.isinf(bmi_f):
                bmi_f = 28.0
        except (TypeError, ValueError):
            bmi_f = 28.0
        bmi_clean = max(10.0, min(80.0, bmi_f))

        try:
            child_f = float(children)
            if np.isnan(child_f) or np.isinf(child_f):
                child_f = 0.0
        except (TypeError, ValueError):
            child_f = 0.0
        children_clean = max(0, min(10, round(child_f)))

        sex_str = str(sex or '').lower().strip()
        sex_clean = 'female' if sex_str in ['female', 'f', 'woman'] else 'male'

        smoker_str = str(smoker or '').lower().strip()
        smoker_clean = 'yes' if smoker_str in ['yes', 'y', 'true', '1', 'smoker'] else 'no'

        region_str = str(region or '').lower().strip().replace(' ', '').replace('-', '')
        valid_regions = {
            'northeast': 'northeast',
            'northwest': 'northwest',
            'southeast': 'southeast',
            'southwest': 'southwest'
        }
        region_clean = valid_regions.get(region_str, 'southwest')

        # Structure input DataFrame with exact schema-aligned feature order
        feature_order = NUMERICAL_COLS + CATEGORICAL_COLS
        input_data = pd.DataFrame([{
            'age': age_clean,
            'bmi': bmi_clean,
            'children': children_clean,
            'sex': sex_clean,
            'smoker': smoker_clean,
            'region': region_clean
        }])[feature_order]

        # Transform using fitted pipeline
        X_trans = self.preprocessor.transform(input_data)

        # Select model (best model or chosen model if available)
        chosen_model = self.model
        if model_name and model_name in self.all_models:
            chosen_model = self.all_models[model_name]

        predicted_annual = float(chosen_model.predict(X_trans)[0])
        predicted_annual = max(800.0, round(predicted_annual, 2))
        predicted_monthly = round(predicted_annual / 12.0, 2)

        # Risk assessment
        risk_score = calculate_heuristic_risk_score(age_clean, bmi_clean, smoker_clean, children_clean)
        risk_category = assign_risk_category(risk_score)
        bmi_cat = assign_bmi_category(bmi_clean)
        age_grp = assign_age_group(age_clean)

        # Feature association explanation (strictly non-causal association wording)
        explanation_insights = self._generate_explanation(age_clean, bmi_clean, smoker_clean, children_clean, region_clean)

        return {
            'age': age_clean,
            'bmi': round(bmi_clean, 2),
            'predicted_annual_cost': predicted_annual,
            'predicted_monthly_cost': predicted_monthly,
            'risk_score': risk_score,
            'risk_category': risk_category,
            'age_category': age_grp,
            'bmi_category': bmi_cat,
            'smoking_status': smoker_clean.capitalize(),
            'dependents': children_clean,
            'region': region_clean.capitalize(),
            'gender': sex_clean.capitalize(),
            'explanation_insights': explanation_insights
        }

    def _generate_explanation(self, age: int, bmi: float, smoker: str, children: int, region: str) -> list:
        """
        Synthesize non-causal statistical associations observed by the predictive model.
        Uses language explicitly stating associations rather than causal guarantees.
        """
        insights = []

        # Smoking association
        if smoker.lower() == 'yes':
            insights.append({
                'feature': 'Smoking Status (Smoker)',
                'impact': 'Strong Upward Association',
                'details': 'In the predictive model, active smoking status is associated with an elevated cost trajectory (+ $14,000 to $30,000+ statistical variance).'
            })
        else:
            insights.append({
                'feature': 'Smoking Status (Non-smoker)',
                'impact': 'Favorable Association',
                'details': 'Non-smoking profiles are associated with the baseline actuarial cost tier.'
            })

        # BMI association
        if bmi >= 30.0:
            compound = " (Compound interaction observed when combined with smoking)" if smoker.lower() == 'yes' else ""
            insights.append({
                'feature': f'BMI ({bmi} - Obese Category)',
                'impact': 'Moderate-to-High Upward Association',
                'details': f'A BMI of {bmi} falls into the obese demographic category, which is associated with increased model cost predictions{compound}.'
            })
        elif bmi >= 25.0:
            insights.append({
                'feature': f'BMI ({bmi} - Overweight Category)',
                'impact': 'Slight Upward Association',
                'details': f'A BMI of {bmi} indicates moderate statistical elevation over baseline normal weight thresholds.'
            })
        else:
            insights.append({
                'feature': f'BMI ({bmi} - Normal/Optimal Range)',
                'impact': 'Neutral/Favorable Association',
                'details': f'BMI values in the 18.5 - 24.9 window reflect standard baseline actuarial parameters.'
            })

        # Age association
        if age >= 50:
            insights.append({
                'feature': f'Age ({age} years)',
                'impact': 'Moderate Upward Association',
                'details': f'Advancing age tiers (50+) correlate with progressive increases in expected actuarial claim expenditures.'
            })
        elif age <= 30:
            insights.append({
                'feature': f'Age ({age} years)',
                'impact': 'Favorable Association',
                'details': f'Early-career demographic age tiers (18-30) reflect lower historical claim frequency curves.'
            })
        else:
            insights.append({
                'feature': f'Age ({age} years)',
                'impact': 'Moderate Trend Progression',
                'details': f'Middle demographic age tiers (31-49) exhibit steady, progressive actuarial trend alignment.'
            })

        # Dependents
        if children > 0:
            insights.append({
                'feature': f'Dependents ({children} Children)',
                'impact': 'Incremental Step Association',
                'details': f'Each dependent adds an incremental risk-pooling adjustment to total household policy expectations.'
            })
        else:
            insights.append({
                'feature': 'Dependents (0 Children)',
                'impact': 'Baseline Individual Policy',
                'details': 'Individual coverage without dependents reflects baseline policy rating with zero child risk-pooling load.'
            })

        # Regional factor
        insights.append({
            'feature': f'Region ({region.capitalize()})',
            'impact': 'Geographic Cost Indexing',
            'details': f'The {region.capitalize()} geographic rating area reflects localized medical network fee schedules and regional provider cost indices.'
        })

        return insights

# Singleton instance helper
_predictor_instance = None

def get_predictor() -> InsurancePredictor:
    """Return cached or new instance of InsurancePredictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = InsurancePredictor()
    return _predictor_instance
