"""
Phase 4 Comprehensive Prediction Engine Independent Test Suite
Validates the complete prediction pipeline independently from the UI:
Inputs -> Validation -> Preprocessing -> Model Inference -> Risk Score -> Explanations
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction.prediction_helper import get_predictor

def test_prediction_engine_scenarios():
    predictor = get_predictor()
    assert predictor.is_loaded, "Predictor must be successfully loaded"

    test_matrix = [
        # 1. Normal valid input
        ("Normal 35yo Non-Smoker", 35, "female", 24.5, 1, "no", "northwest"),
        
        # 2. Minimum reasonable values
        ("Minimum Boundary (18yo, BMI 15.0, 0 kids)", 18, "male", 15.0, 0, "no", "northeast"),
        
        # 3. Maximum reasonable values
        ("Maximum Boundary (85yo, BMI 55.0, 5 kids)", 85, "male", 55.0, 5, "yes", "southeast"),
        
        # 4. Different age groups
        ("Age Group: 18-29 (Young adult)", 22, "female", 22.0, 0, "no", "southwest"),
        ("Age Group: 30-39 (Career)", 34, "male", 26.0, 1, "no", "southwest"),
        ("Age Group: 40-49 (Mid-career)", 44, "female", 28.0, 2, "no", "northwest"),
        ("Age Group: 50+ (Senior)", 58, "male", 29.0, 0, "no", "northeast"),
        
        # 5. Different BMI ranges
        ("BMI Range: Underweight (BMI 16.5)", 28, "female", 16.5, 0, "no", "southeast"),
        ("BMI Range: Normal (BMI 22.5)", 28, "male", 22.5, 0, "no", "southeast"),
        ("BMI Range: Overweight (BMI 27.5)", 28, "female", 27.5, 0, "no", "southeast"),
        ("BMI Range: Class I Obese (BMI 32.5)", 28, "male", 32.5, 0, "no", "southeast"),
        ("BMI Range: Morbidly Obese (BMI 44.0)", 28, "female", 44.0, 0, "no", "southeast"),
        
        # 6. Smoker vs Non-Smoker comparison (identical otherwise)
        ("Smoking Pair: Non-Smoker", 42, "male", 31.0, 2, "no", "southwest"),
        ("Smoking Pair: Smoker", 42, "male", 31.0, 2, "yes", "southwest"),
        
        # 7. Different Dependents
        ("Dependents: 0 Children", 35, "female", 25.0, 0, "no", "northeast"),
        ("Dependents: 1 Child", 35, "female", 25.0, 1, "no", "northeast"),
        ("Dependents: 3 Children", 35, "female", 25.0, 3, "no", "northeast"),
        ("Dependents: 5 Children", 35, "female", 25.0, 5, "no", "northeast"),
        
        # 8. All Geographic Regions (identical otherwise)
        ("Region: Northeast", 40, "female", 26.0, 1, "no", "northeast"),
        ("Region: Northwest", 40, "female", 26.0, 1, "no", "northwest"),
        ("Region: Southeast", 40, "female", 26.0, 1, "no", "southeast"),
        ("Region: Southwest", 40, "female", 26.0, 1, "no", "southwest"),
        
        # 9. Lifestyle & Sanitization Inputs
        ("Dirty Strings: Upper, whitespace, boolean smoker", " 38 ", " FEMALE ", "29.2", "2", "TRUE", " North-East "),
    ]

    print(f"{'Test Case Description':<44} | {'Annual ($)':<12} | {'Monthly ($)':<11} | {'Score':<6} | {'Tier':<6} | {'Status'}")
    print("=" * 95)

    all_results = []
    for desc, age, sex, bmi, children, smoker, region in test_matrix:
        res = predictor.predict_single(
            age=age, sex=sex, bmi=bmi, children=children, smoker=smoker, region=region
        )
        
        # Assertions
        assert not np.isnan(res['predicted_annual_cost']), f"NaN cost in {desc}"
        assert not np.isinf(res['predicted_annual_cost']), f"Inf cost in {desc}"
        assert res['predicted_annual_cost'] > 0, f"Negative/zero cost in {desc}"
        assert res['predicted_monthly_cost'] == round(res['predicted_annual_cost'] / 12.0, 2)
        assert 0.0 <= res['risk_score'] <= 100.0, f"Risk score out of bounds in {desc}"
        assert res['risk_category'] in ['LOW', 'MEDIUM', 'HIGH'], f"Invalid risk category in {desc}"
        assert len(res['explanation_insights']) >= 3, f"Insufficient explanation insights in {desc}"

        all_results.append((desc, res))
        print(f"{desc:<44} | ${res['predicted_annual_cost']:>10,.2f} | ${res['predicted_monthly_cost']:>9,.2f} | {res['risk_score']:>5.1f} | {res['risk_category']:<6} | PASS")

    # All assertions passed cleanly

if __name__ == '__main__':
    test_prediction_engine_scenarios()
