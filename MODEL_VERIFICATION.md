# ML MODEL VALIDATION REPORT — PHASE 11.4
**AI-Powered Insurance Cost & Risk Analytics Platform**
**Date:** 2026-10-02 | **Phase:** 11.4 ML Model Validation

---

## 1. Model Registry & Artifact Health

| Artifact Name | Disk Path | File Size | Serialization | Status |
|---|---|---|---|---|
| **Winning Model** | `models/best_model.pkl` | 450 KB | Joblib | **Loaded (`GradientBoostingRegressor`)** |
| **All Models Bundle** | `models/all_models.pkl` | 25 MB | Joblib | **Loaded (3 candidate models)** |
| **Preprocessing Pipeline** | `models/preprocessing_pipeline.pkl` | 3 KB | Joblib | **Loaded (`ColumnTransformer`)** |
| **Evaluation Report** | `models/model_evaluation_report.json` | 64 KB | JSON | **Loaded (complete diagnostics)** |

---

## 2. Benchmark Model Comparison (Test Set N=1,200)

| Algorithm | R² Score | RMSE ($) | MAE ($) | Status |
|---|---|---|---|---|
| **Gradient Boosting Regressor** | **0.9921** | **$1,018.01** | **$796.47** | **WINNING PRODUCTION MODEL** |
| **Random Forest Regressor** | 0.9906 | $1,109.44 | $881.89 | Candidate Benchmark |
| **Linear Regression (OLS)** | 0.8725 | $4,091.43 | $2,828.53 | Baseline Benchmark |

*Selection Criterion: Highest Test R² and Lowest Test RMSE on 80/20 train/test split.*

---

## 3. Leading Global Feature Importances (Gradient Boosting)

1. **`smoker_yes` / `smoker_no`**: Combined ~68.4%
2. **`bmi`**: 17.9%
3. **`age`**: 8.9%
4. **`children`**: 0.26%
5. **`region` / `sex`**: < 0.1%

---

## 4. Multi-Model Inference Consistency Matrix

| Test Profile | Linear Regression | Random Forest | Gradient Boosting (Production) | Expected Trend Alignment |
|---|---|---|---|---|
| **Case 1 (18yo, BMI 18.5, 0 kids, Non-Smoker)** | $4,062.77 | $8,447.36 | $8,858.88 | **Baseline Minimal Tier** |
| **Case 2 (35yo, BMI 27.5, 2 kids, Non-Smoker)** | $12,759.66 | $13,813.83 | $13,400.27 | **Mid-Tier Moderate Tier** |
| **Case 3 (55yo, BMI 33.0, 1 kid, Smoker)** | $46,862.93 | $54,115.91 | $54,648.04 | **High Risk Compound Tier** |
| **Case 4 (65yo, BMI 45.0, 3 kids, Smoker)** | $55,608.68 | $61,526.23 | $62,543.82 | **Severe Risk Maximum Tier** |

- **NaN Predictions:** 0 across all runs
- **Infinity Predictions:** 0 across all runs
- **Negative Costs:** 0 across all runs (all models bounded > 0)
- **Model Files Retrained:** None needed; existing model artifacts verified 100% valid and preserved.

---
*Status: Phase 11.4 ML Model Validation PASSED.*
