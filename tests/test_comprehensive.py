"""
Phase 9 — Comprehensive Automated Test Suite
Covers all 9 required categories:

  1.  Data loading              — file integrity, column schema, dtypes, value ranges, duplicates
  2.  Preprocessing             — unit tests for each helper function, vectorized/scalar consistency
  3.  Feature validation        — schema alignment between processed data and trained model
  4.  Model loading             — artifact existence, sklearn types, feature counts, all_models keys
  5.  Prediction                — monotonicity, cost floor, NULL/Inf guards, all regions
  6.  Risk calculation          — exact anchoring values, boundary thresholds, compound interactions
  7.  Scenario analysis         — baseline delta invariant, directional cost ordering
  8.  Dashboard helper funcs    — compute_kpis keys, apply_chart_style, missing-column guard paths
  9.  Edge cases                — single-row KPIs, empty feature importances, partial DataFrame

DO NOT delete or modify existing tests.
All assertions are precise — no weakened conditions used to force passing.
"""

import os
import sys
import json
import joblib
import math

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processing.preprocess import (
    assign_bmi_category,
    assign_age_group,
    calculate_heuristic_risk_score,
    assign_risk_category,
    engineer_features,
    _vectorized_risk_score,
    build_preprocessing_pipeline,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
)
from src.prediction.prediction_helper import get_predictor
from src.analytics.analytics_helper import (
    apply_chart_style,
    compute_kpis,
    create_cost_by_age_group_chart,
    create_cost_by_bmi_category_chart,
    create_risk_distribution_chart,
    create_risk_gauge_chart,
    create_feature_importance_chart,
    create_scenario_comparison_chart,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RAW_PATH   = os.path.join(PROJECT_ROOT, "data", "raw",       "insurance_dataset.csv")
PROC_PATH  = os.path.join(PROJECT_ROOT, "data", "processed", "insurance_processed.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def _load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)


def _load_proc() -> pd.DataFrame:
    return pd.read_csv(PROC_PATH)


# ===========================================================================
# CATEGORY 1 — DATA LOADING
# ===========================================================================

class TestDataLoading:
    """Validate raw and processed dataset integrity."""

    def test_raw_file_exists(self):
        assert os.path.exists(RAW_PATH), f"Raw dataset missing: {RAW_PATH}"

    def test_processed_file_exists(self):
        assert os.path.exists(PROC_PATH), f"Processed dataset missing: {PROC_PATH}"

    def test_raw_record_count(self):
        df = _load_raw()
        assert len(df) >= 5000, f"Expected >= 5000 raw records, got {len(df)}"

    def test_processed_record_count(self):
        df = _load_proc()
        assert len(df) >= 5000, f"Expected >= 5000 processed records, got {len(df)}"

    def test_raw_required_columns(self):
        df = _load_raw()
        required = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
        for col in required:
            assert col in df.columns, f"Missing column '{col}' in raw data"

    def test_processed_engineered_columns(self):
        df = _load_proc()
        for col in ["bmi_category", "age_group", "risk_score", "risk_category"]:
            assert col in df.columns, f"Missing engineered column '{col}' in processed data"

    def test_raw_no_nulls(self):
        df = _load_raw()
        null_count = df[["age", "sex", "bmi", "children", "smoker", "region", "charges"]].isnull().sum().sum()
        assert null_count == 0, f"Raw dataset contains {null_count} null values in core columns"

    def test_processed_no_nulls(self):
        df = _load_proc()
        assert df.isnull().sum().sum() == 0, "Processed dataset must have zero null values"

    def test_raw_age_range(self):
        df = _load_raw()
        assert df["age"].min() >= 18, "Minimum age must be >= 18"
        assert df["age"].max() <= 100, "Maximum age must be <= 100"

    def test_raw_bmi_range(self):
        df = _load_raw()
        assert df["bmi"].min() > 0, "BMI must be > 0"
        assert df["bmi"].max() < 200, "BMI implausibly high (> 200)"

    def test_raw_charges_positive(self):
        df = _load_raw()
        assert (df["charges"] > 0).all(), "All insurance charges must be positive"

    def test_raw_children_non_negative(self):
        df = _load_raw()
        assert (df["children"] >= 0).all(), "Children count must be >= 0"

    def test_raw_sex_domain(self):
        df = _load_raw()
        assert set(df["sex"].unique()).issubset({"male", "female"}), \
            f"Unexpected sex values: {df['sex'].unique()}"

    def test_raw_smoker_domain(self):
        df = _load_raw()
        assert set(df["smoker"].unique()).issubset({"yes", "no"}), \
            f"Unexpected smoker values: {df['smoker'].unique()}"

    def test_raw_region_domain(self):
        df = _load_raw()
        expected = {"northeast", "northwest", "southeast", "southwest"}
        assert set(df["region"].unique()).issubset(expected), \
            f"Unexpected region values: {df['region'].unique()}"

    def test_raw_no_excessive_duplicates(self):
        df = _load_raw()
        dup_rate = df.duplicated().sum() / len(df)
        # Allow small synthetic overlap but flag > 5%
        assert dup_rate < 0.05, \
            f"Duplicate rate {dup_rate:.1%} exceeds 5% threshold"

    def test_processed_risk_category_domain(self):
        df = _load_proc()
        assert set(df["risk_category"].unique()).issubset({"LOW", "MEDIUM", "HIGH"}), \
            f"Unexpected risk_category values: {df['risk_category'].unique()}"

    def test_processed_risk_score_range(self):
        df = _load_proc()
        assert df["risk_score"].between(0.0, 100.0).all(), \
            "All risk scores must be in [0.0, 100.0]"


# ===========================================================================
# CATEGORY 2 — PREPROCESSING
# ===========================================================================

class TestPreprocessing:
    """Unit tests for every preprocessing helper function."""

    @pytest.mark.parametrize("bmi,expected", [
        (14.0,  "Underweight"),
        (18.4,  "Underweight"),
        (18.5,  "Normal"),
        (22.0,  "Normal"),
        (24.9,  "Normal"),
        (25.0,  "Overweight"),
        (27.5,  "Overweight"),
        (29.9,  "Overweight"),
        (30.0,  "Obese"),
        (35.0,  "Obese"),
        (55.0,  "Obese"),
    ])
    def test_assign_bmi_category(self, bmi, expected):
        result = assign_bmi_category(bmi)
        assert result == expected, f"BMI {bmi} -> expected '{expected}', got '{result}'"

    @pytest.mark.parametrize("age,expected", [
        (18,  "18-29"),
        (25,  "18-29"),
        (29,  "18-29"),
        (30,  "30-39"),
        (35,  "30-39"),
        (39,  "30-39"),
        (40,  "40-49"),
        (45,  "40-49"),
        (49,  "40-49"),
        (50,  "50+"),
        (65,  "50+"),
        (85,  "50+"),
    ])
    def test_assign_age_group(self, age, expected):
        result = assign_age_group(age)
        assert result == expected, f"Age {age} -> expected '{expected}', got '{result}'"

    @pytest.mark.parametrize("score,expected", [
        (0.0,   "LOW"),
        (17.5,  "LOW"),
        (34.9,  "LOW"),
        (35.0,  "MEDIUM"),
        (50.0,  "MEDIUM"),
        (69.9,  "MEDIUM"),
        (70.0,  "HIGH"),
        (85.0,  "HIGH"),
        (100.0, "HIGH"),
    ])
    def test_assign_risk_category(self, score, expected):
        result = assign_risk_category(score)
        assert result == expected, f"Score {score} -> expected '{expected}', got '{result}'"

    def test_risk_score_baseline_non_smoker_young_normal(self):
        """
        Young (18), normal BMI (22.0), non-smoker, 0 children.
        Normal BMI (18.5-24.9) adds 0 pts (no BMI tier matched).
        Expected: baseline(5) + bmi_normal(0) + age_pts(0) + children(0) = 5.0
        """
        score = calculate_heuristic_risk_score(18, 22.0, "no", 0)
        assert score == 5.0, f"Expected 5.0, got {score}"

    def test_risk_score_pure_baseline_underweight(self):
        """Age 18, underweight BMI (15.0), non-smoker, 0 children = 10.0."""
        score = calculate_heuristic_risk_score(18, 15.0, "no", 0)
        # baseline 5 + underweight 5 + age_pts 0 + children 0 = 10.0
        assert score == 10.0, f"Expected 10.0, got {score}"

    def test_risk_score_smoker_delta_is_48(self):
        """Smoking adds exactly 48 pts over the identical non-smoker profile."""
        non_smoker = calculate_heuristic_risk_score(30, 22.0, "no", 0)
        smoker     = calculate_heuristic_risk_score(30, 22.0, "yes", 0)
        assert smoker - non_smoker == 48.0, \
            f"Smoker delta should be 48.0, got {smoker - non_smoker}"

    def test_risk_score_obese_smoker_compound_bonus(self):
        """
        Obese + smoker triggers a +10 compound bonus in addition to the BMI obesity tier.
        Lean smoker (BMI 22, Normal tier = +0 BMI pts): baseline(5)+smoke(48)+bmi(0)+age_30(~4.6) = ~57.6
        Obese smoker (BMI 32, >= 30 tier = +20 BMI pts + compound +10): 57.6 + 20 + 10 = ~87.6
        Delta = 87.6 - 57.6 = 30.0
        """
        smoker_lean  = calculate_heuristic_risk_score(30, 22.0, "yes", 0)
        smoker_obese = calculate_heuristic_risk_score(30, 32.0, "yes", 0)
        delta = round(smoker_obese - smoker_lean, 1)
        assert delta == 30.0, \
            f"Obese-smoker compound delta should be 30.0, got {delta}"

    def test_risk_score_clipped_to_100(self):
        score = calculate_heuristic_risk_score(100, 50.0, "yes", 10)
        assert score == 100.0, f"Score must be clipped to 100.0, got {score}"

    def test_risk_score_never_negative(self):
        score = calculate_heuristic_risk_score(18, 10.0, "no", 0)
        assert score >= 0.0

    def test_risk_score_case_insensitive_smoker(self):
        lower = calculate_heuristic_risk_score(30, 22.0, "yes", 0)
        upper = calculate_heuristic_risk_score(30, 22.0, "YES", 0)
        mixed = calculate_heuristic_risk_score(30, 22.0, "Yes", 0)
        assert lower == upper == mixed, "Smoker matching must be case-insensitive"

    def test_vectorized_matches_scalar(self):
        """_vectorized_risk_score must produce results identical to the scalar function."""
        rows = [
            {"age": 18, "bmi": 22.0, "smoker": "no",  "children": 0},
            {"age": 45, "bmi": 32.0, "smoker": "yes", "children": 2},
            {"age": 62, "bmi": 40.0, "smoker": "yes", "children": 3},
            {"age": 25, "bmi": 15.0, "smoker": "no",  "children": 0},
        ]
        df = pd.DataFrame(rows)
        vec_scores = _vectorized_risk_score(df).tolist()
        for i, row in enumerate(rows):
            scalar = calculate_heuristic_risk_score(
                float(row["age"]), float(row["bmi"]), str(row["smoker"]), int(row["children"])
            )
            assert abs(vec_scores[i] - scalar) < 0.01, \
                f"Row {i}: vectorized={vec_scores[i]:.1f}, scalar={scalar:.1f}"

    def test_engineer_features_adds_columns(self):
        df_raw = _load_raw().head(50)
        df_eng = engineer_features(df_raw)
        for col in ["bmi_category", "age_group", "risk_score", "risk_category"]:
            assert col in df_eng.columns

    def test_engineer_features_preserves_row_count(self):
        df_raw = _load_raw().head(100)
        df_eng = engineer_features(df_raw)
        assert len(df_eng) == len(df_raw)

    def test_engineer_features_does_not_mutate_input(self):
        df_raw = _load_raw().head(10)
        original_cols = list(df_raw.columns)
        _ = engineer_features(df_raw)
        assert list(df_raw.columns) == original_cols, \
            "engineer_features must not mutate the input DataFrame"

    def test_build_preprocessing_pipeline_is_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        pp = build_preprocessing_pipeline()
        assert isinstance(pp, ColumnTransformer)

    def test_preprocessing_pipeline_output_shape(self):
        df = _load_proc()
        X = df[NUMERICAL_COLS + CATEGORICAL_COLS].head(200)
        pp = build_preprocessing_pipeline()
        X_trans = pp.fit_transform(X)
        # 3 numerical + OHE(sex:2, smoker:2, region:4) = 11 features
        assert X_trans.shape == (200, 11), \
            f"Expected shape (200, 11), got {X_trans.shape}"

    def test_preprocessing_pipeline_no_nan_output(self):
        df = _load_proc()
        X = df[NUMERICAL_COLS + CATEGORICAL_COLS].head(100)
        pp = build_preprocessing_pipeline()
        X_trans = pp.fit_transform(X)
        X_dense = X_trans.toarray() if hasattr(X_trans, "toarray") else np.asarray(X_trans)
        assert not np.isnan(X_dense).any(), "Pipeline output must contain no NaN values"


# ===========================================================================
# CATEGORY 3 — FEATURE VALIDATION
# ===========================================================================

class TestFeatureValidation:
    """Validate alignment between processed data schema and trained model."""

    def test_model_feature_count_matches_pipeline(self):
        best_model   = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl"))
        df = _load_proc()
        X_sample = df[NUMERICAL_COLS + CATEGORICAL_COLS].head(5)
        X_trans  = preprocessor.transform(X_sample)
        assert X_trans.shape[1] == best_model.n_features_in_, (
            f"Model expects {best_model.n_features_in_} features "
            f"but pipeline produces {X_trans.shape[1]}"
        )

    def test_feature_names_report_matches_pipeline(self):
        with open(os.path.join(MODELS_DIR, "model_evaluation_report.json")) as f:
            report = json.load(f)
        preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl"))
        df = _load_proc()
        preprocessor.transform(df[NUMERICAL_COLS + CATEGORICAL_COLS].head(5))
        pipeline_names = set(
            n.replace("num__", "").replace("cat__", "")
            for n in preprocessor.get_feature_names_out()
        )
        report_names = set(report.get("feature_names", []))
        assert pipeline_names == report_names, \
            f"Pipeline names {pipeline_names} don't match report names {report_names}"

    def test_numerical_cols_schema_unchanged(self):
        assert NUMERICAL_COLS == ["age", "bmi", "children"]

    def test_categorical_cols_schema_unchanged(self):
        assert CATEGORICAL_COLS == ["sex", "smoker", "region"]

    def test_target_col_schema_unchanged(self):
        assert TARGET_COL == "charges"

    def test_inference_feature_order_accepted_by_model(self):
        """A single inference pass must not raise a shape/feature mismatch error."""
        predictor = get_predictor()
        result = predictor.predict_single(35, "female", 26.0, 1, "no", "northeast")
        assert result["predicted_annual_cost"] > 0


# ===========================================================================
# CATEGORY 4 — MODEL LOADING
# ===========================================================================

class TestModelLoading:
    """Validate all model artifacts load correctly with expected properties."""

    def test_best_model_pkl_exists(self):
        assert os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl"))

    def test_preprocessing_pipeline_pkl_exists(self):
        assert os.path.exists(os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl"))

    def test_all_models_pkl_exists(self):
        assert os.path.exists(os.path.join(MODELS_DIR, "all_models.pkl"))

    def test_evaluation_report_json_exists(self):
        assert os.path.exists(os.path.join(MODELS_DIR, "model_evaluation_report.json"))

    def test_all_three_models_present(self):
        all_models = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))
        expected = {
            "Linear Regression",
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
        }
        assert set(all_models.keys()) == expected

    def test_best_model_name_in_report(self):
        with open(os.path.join(MODELS_DIR, "model_evaluation_report.json")) as f:
            report = json.load(f)
        assert report["best_model_name"] == "Gradient Boosting Regressor"

    def test_all_models_have_predict_method(self):
        all_models = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))
        for name, model in all_models.items():
            assert hasattr(model, "predict"), f"'{name}' lacks .predict()"

    def test_all_models_are_fitted(self):
        all_models = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))
        for name, model in all_models.items():
            assert hasattr(model, "n_features_in_"), \
                f"'{name}' does not appear fitted (missing n_features_in_)"

    def test_all_models_accept_11_features(self):
        all_models = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))
        for name, model in all_models.items():
            assert model.n_features_in_ == 11, \
                f"'{name}' expects {model.n_features_in_} features, not 11"

    def test_evaluation_report_required_keys(self):
        with open(os.path.join(MODELS_DIR, "model_evaluation_report.json")) as f:
            report = json.load(f)
        for key in ["best_model_name", "models", "y_true_sample",
                    "feature_names", "training_samples", "test_samples"]:
            assert key in report, f"Report missing key '{key}'"

    def test_evaluation_report_metrics_keys_per_model(self):
        with open(os.path.join(MODELS_DIR, "model_evaluation_report.json")) as f:
            report = json.load(f)
        for model_name, model_data in report["models"].items():
            for metric in ["R2", "RMSE", "MAE", "MSE"]:
                assert metric in model_data["metrics"], \
                    f"'{model_name}' missing metric '{metric}'"

    def test_gradient_boosting_r2_above_threshold(self):
        """Gradient Boosting champion must have R² >= 0.95."""
        with open(os.path.join(MODELS_DIR, "model_evaluation_report.json")) as f:
            report = json.load(f)
        r2 = report["models"]["Gradient Boosting Regressor"]["metrics"]["R2"]
        assert r2 >= 0.95, f"Gradient Boosting R² = {r2} is below 0.95 threshold"

    def test_predictor_singleton_reuse(self):
        p1 = get_predictor()
        p2 = get_predictor()
        assert p1 is p2, "get_predictor() must return the same singleton instance"


# ===========================================================================
# CATEGORY 5 — PREDICTION
# ===========================================================================

class TestPrediction:
    """Validate prediction correctness, monotonicity, and output schema."""

    @pytest.fixture(scope="class")
    @classmethod
    def predictor(cls):
        return get_predictor()

    def test_predictor_is_loaded(self, predictor):
        assert predictor.is_loaded

    def test_output_schema_all_keys(self, predictor):
        result = predictor.predict_single(35, "female", 26.0, 1, "no", "northeast")
        required_keys = [
            "age", "bmi", "predicted_annual_cost", "predicted_monthly_cost",
            "risk_score", "risk_category", "age_category", "bmi_category",
            "smoking_status", "dependents", "region", "gender", "explanation_insights",
        ]
        for k in required_keys:
            assert k in result, f"Prediction result missing key '{k}'"

    def test_monthly_equals_annual_over_12(self, predictor):
        result = predictor.predict_single(40, "male", 29.0, 2, "no", "southwest")
        expected = round(result["predicted_annual_cost"] / 12.0, 2)
        assert result["predicted_monthly_cost"] == expected

    def test_cost_floor_enforced(self, predictor):
        result = predictor.predict_single(18, "female", 15.0, 0, "no", "southwest")
        assert result["predicted_annual_cost"] >= 800.0, \
            f"Cost floor violated: ${result['predicted_annual_cost']:.2f}"

    def test_smoker_monotonicity(self, predictor):
        non_smoker = predictor.predict_single(40, "male", 28.0, 1, "no",  "northeast")
        smoker     = predictor.predict_single(40, "male", 28.0, 1, "yes", "northeast")
        assert smoker["predicted_annual_cost"] > non_smoker["predicted_annual_cost"]

    def test_age_monotonicity_non_smokers(self, predictor):
        young  = predictor.predict_single(22, "female", 23.0, 0, "no", "northwest")
        senior = predictor.predict_single(60, "female", 23.0, 0, "no", "northwest")
        assert senior["predicted_annual_cost"] > young["predicted_annual_cost"]

    def test_obese_costs_more_than_normal(self, predictor):
        normal = predictor.predict_single(40, "male", 22.0, 0, "no", "southeast")
        obese  = predictor.predict_single(40, "male", 35.0, 0, "no", "southeast")
        assert obese["predicted_annual_cost"] > normal["predicted_annual_cost"]

    def test_all_four_regions_valid(self, predictor):
        for region in ["northeast", "northwest", "southeast", "southwest"]:
            result = predictor.predict_single(35, "male", 26.0, 1, "no", region)
            assert result["predicted_annual_cost"] > 0
            assert result["region"] == region.capitalize()

    def test_no_nan_in_output(self, predictor):
        result = predictor.predict_single(45, "female", 30.0, 2, "yes", "southeast")
        assert not math.isnan(result["predicted_annual_cost"])
        assert not math.isnan(result["predicted_monthly_cost"])
        assert not math.isnan(result["risk_score"])

    def test_no_inf_in_output(self, predictor):
        result = predictor.predict_single(45, "female", 30.0, 2, "yes", "southeast")
        assert not math.isinf(result["predicted_annual_cost"])

    def test_explanation_insights_count(self, predictor):
        result = predictor.predict_single(35, "male", 26.0, 1, "no", "southwest")
        assert len(result["explanation_insights"]) >= 3

    def test_explanation_insights_keys(self, predictor):
        result = predictor.predict_single(35, "male", 26.0, 1, "no", "southwest")
        for item in result["explanation_insights"]:
            for key in ["feature", "impact", "details"]:
                assert key in item, f"Insight missing key '{key}'"


# ===========================================================================
# CATEGORY 6 — RISK CALCULATION
# ===========================================================================

class TestRiskCalculation:
    """Exact numerical anchoring and threshold boundary tests."""

    def test_low_boundary_exact_349(self):
        assert assign_risk_category(34.9) == "LOW"

    def test_medium_starts_at_350(self):
        assert assign_risk_category(35.0) == "MEDIUM"

    def test_medium_boundary_exact_699(self):
        assert assign_risk_category(69.9) == "MEDIUM"

    def test_high_starts_at_700(self):
        assert assign_risk_category(70.0) == "HIGH"

    def test_young_healthy_non_smoker_is_low(self):
        predictor = get_predictor()
        result = predictor.predict_single(22, "female", 21.0, 0, "no", "southwest")
        assert result["risk_category"] == "LOW"

    def test_obese_senior_smoker_is_high(self):
        predictor = get_predictor()
        result = predictor.predict_single(50, "male", 35.0, 2, "yes", "southeast")
        assert result["risk_category"] == "HIGH"

    def test_all_processed_risk_scores_in_range(self):
        df = _load_proc()
        out_of_range = df[(df["risk_score"] < 0) | (df["risk_score"] > 100)]
        assert len(out_of_range) == 0, \
            f"{len(out_of_range)} rows have risk_score outside [0, 100]"

    def test_risk_category_consistent_with_risk_score_sample(self):
        df = _load_proc()
        for _, row in df.sample(200, random_state=99).iterrows():
            expected = assign_risk_category(row["risk_score"])
            assert row["risk_category"] == expected, (
                f"Mismatch: score={row['risk_score']}, "
                f"stored={row['risk_category']}, expected={expected}"
            )

    def test_smoker_always_higher_risk_score(self):
        profiles = [(22, 20.0, 0), (35, 26.0, 1), (55, 33.0, 3)]
        for age, bmi, children in profiles:
            ns = calculate_heuristic_risk_score(age, bmi, "no",  children)
            sm = calculate_heuristic_risk_score(age, bmi, "yes", children)
            assert sm > ns, f"Smoker score not higher for age={age}, bmi={bmi}"

    def test_obese_bmi_higher_risk_score_than_normal(self):
        normal = calculate_heuristic_risk_score(35, 22.0, "no", 0)
        obese  = calculate_heuristic_risk_score(35, 32.0, "no", 0)
        assert obese > normal


# ===========================================================================
# CATEGORY 7 — SCENARIO ANALYSIS
# ===========================================================================

class TestScenarioAnalysis:
    """Test what-if scenario delta calculations and ordering."""

    @pytest.fixture(scope="class")
    @classmethod
    def predictor(cls):
        return get_predictor()

    def _predict_all(self, predictor, scenarios):
        results = []
        for s in scenarios:
            pred = predictor.predict_single(
                age=s["age"], sex=s["sex"], bmi=s["bmi"],
                children=s["children"], smoker=s["smoker"], region=s["region"],
            )
            results.append({
                "Scenario": s["name"],
                "Predicted Annual Cost": pred["predicted_annual_cost"],
                "Monthly Cost": pred["predicted_monthly_cost"],
                "Risk Score": pred["risk_score"],
                "Risk Category": pred["risk_category"],
            })
        return pd.DataFrame(results)

    def test_baseline_delta_is_zero(self, predictor):
        scenarios = [
            {"name": "A", "age": 25, "sex": "female", "bmi": 22.0,
             "children": 0, "smoker": "no", "region": "southwest"},
            {"name": "B", "age": 45, "sex": "male",   "bmi": 30.0,
             "children": 2, "smoker": "yes", "region": "southeast"},
        ]
        df = self._predict_all(predictor, scenarios)
        baseline = df.loc[0, "Predicted Annual Cost"]
        df["delta"] = df["Predicted Annual Cost"] - baseline
        assert df.loc[0, "delta"] == 0.0

    def test_high_risk_scenario_costs_more(self, predictor):
        low  = predictor.predict_single(22, "female", 21.0, 0, "no",  "southwest")
        high = predictor.predict_single(62, "male",   38.0, 3, "yes", "southeast")
        assert high["predicted_annual_cost"] > low["predicted_annual_cost"]

    def test_four_scenario_cost_ordering(self, predictor):
        base   = predictor.predict_single(25, "female", 21.0, 0, "no",  "southwest")
        s_obese = predictor.predict_single(25, "female", 35.0, 0, "no",  "southwest")
        s_all   = predictor.predict_single(60, "female", 35.0, 0, "yes", "southwest")
        assert s_obese["predicted_annual_cost"] > base["predicted_annual_cost"]
        assert s_all["predicted_annual_cost"]   > s_obese["predicted_annual_cost"]

    def test_scenario_csv_serialisation(self, predictor):
        scenarios = [
            {"name": "A", "age": 25, "sex": "female", "bmi": 22.0,
             "children": 0, "smoker": "no", "region": "southwest"},
            {"name": "B", "age": 50, "sex": "male",   "bmi": 33.0,
             "children": 2, "smoker": "yes", "region": "northeast"},
        ]
        df = self._predict_all(predictor, scenarios)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        assert len(csv_bytes) > 100

    def test_positive_pct_difference_direction(self, predictor):
        cheap     = predictor.predict_single(22, "female", 21.0, 0, "no",  "southwest")
        expensive = predictor.predict_single(60, "male",   36.0, 3, "yes", "southeast")
        pct = (
            (expensive["predicted_annual_cost"] - cheap["predicted_annual_cost"])
            / cheap["predicted_annual_cost"] * 100.0
        )
        assert pct > 0.0


# ===========================================================================
# CATEGORY 8 — DASHBOARD HELPER FUNCTIONS
# ===========================================================================

class TestDashboardHelpers:
    """Unit tests for analytics_helper module functions."""

    @pytest.fixture(scope="class")
    @classmethod
    def df(cls):
        return _load_proc()

    def test_compute_kpis_required_keys(self, df):
        kpis = compute_kpis(df)
        for k in ["total_customers", "avg_cost", "median_cost", "min_cost",
                  "max_cost", "avg_age", "avg_bmi", "high_risk_pct"]:
            assert k in kpis, f"compute_kpis missing key '{k}'"

    def test_compute_kpis_total_matches_len(self, df):
        kpis = compute_kpis(df)
        assert kpis["total_customers"] == len(df)

    def test_compute_kpis_min_le_median_le_max(self, df):
        kpis = compute_kpis(df)
        assert kpis["min_cost"] <= kpis["median_cost"] <= kpis["max_cost"]

    def test_compute_kpis_high_risk_pct_in_range(self, df):
        kpis = compute_kpis(df)
        assert 0.0 <= kpis["high_risk_pct"] <= 100.0

    def test_compute_kpis_empty_df_zeroes(self):
        df_empty = pd.DataFrame(
            columns=["age", "sex", "bmi", "children", "smoker",
                     "region", "charges", "risk_category"]
        )
        kpis = compute_kpis(df_empty)
        assert kpis["total_customers"] == 0
        assert kpis["avg_cost"] == 0.0
        assert kpis["high_risk_pct"] == 0.0

    def test_apply_chart_style_returns_figure(self, df):
        import plotly.express as px
        fig = px.histogram(df, x="charges")
        styled = apply_chart_style(fig, "Test Chart", height=400)
        assert styled is not None
        assert styled.layout.height == 400

    def test_apply_chart_style_sets_title(self, df):
        import plotly.express as px
        fig = px.histogram(df, x="charges")
        styled = apply_chart_style(fig, "MyTitle")
        assert "MyTitle" in styled.layout.title.text

    def test_create_risk_distribution_chart_missing_column(self):
        """Must return an empty Figure (not raise) when risk_category is absent."""
        df_no_cat = _load_proc().drop(columns=["risk_category"])
        result = create_risk_distribution_chart(df_no_cat)
        assert isinstance(result, go.Figure)

    def test_create_feature_importance_chart_empty_dict(self):
        """Empty dict must not raise."""
        result = create_feature_importance_chart({}, "Empty Importances")
        assert isinstance(result, go.Figure)

    def test_create_scenario_comparison_chart_structure(self):
        df_sc = pd.DataFrame([
            {"Scenario": "A", "Predicted Annual Cost": 8000.0},
            {"Scenario": "B", "Predicted Annual Cost": 45000.0},
        ])
        fig = create_scenario_comparison_chart(df_sc)
        assert isinstance(fig, go.Figure)

    def test_create_risk_gauge_all_tiers(self):
        for score, cat in [(10.0, "LOW"), (52.0, "MEDIUM"), (85.0, "HIGH")]:
            fig = create_risk_gauge_chart(score, cat)
            assert isinstance(fig, go.Figure)

    def test_create_cost_by_bmi_category_chart_subset(self):
        """Chart must work on a filtered subset with only 2 BMI categories."""
        df_sub = _load_proc()
        df_sub = df_sub[df_sub["bmi_category"].isin(["Normal", "Obese"])]
        fig = create_cost_by_bmi_category_chart(df_sub)
        assert isinstance(fig, go.Figure)

    def test_create_cost_by_age_group_chart_returns_figure(self, df):
        fig = create_cost_by_age_group_chart(df)
        assert isinstance(fig, go.Figure)


# ===========================================================================
# CATEGORY 9 — ADDITIONAL EDGE CASES
# ===========================================================================

class TestAdditionalEdgeCases:
    """Edge cases not covered by the existing test suite."""

    def test_compute_kpis_single_row(self):
        df_single = _load_proc().iloc[:1]
        kpis = compute_kpis(df_single)
        assert kpis["min_cost"] == kpis["max_cost"] == kpis["avg_cost"]
        assert kpis["total_customers"] == 1

    def test_prediction_max_children(self):
        predictor = get_predictor()
        result = predictor.predict_single(35, "male", 26.0, 10, "no", "northeast")
        assert result["dependents"] == 10
        assert result["predicted_annual_cost"] > 0

    def test_bmi_boundary_185_is_normal(self):
        assert assign_bmi_category(18.5) == "Normal"

    def test_bmi_boundary_250_is_overweight(self):
        assert assign_bmi_category(25.0) == "Overweight"

    def test_bmi_boundary_300_is_obese(self):
        assert assign_bmi_category(30.0) == "Obese"

    def test_age_group_boundary_30_is_30_39(self):
        assert assign_age_group(30) == "30-39"

    def test_age_group_boundary_40_is_40_49(self):
        assert assign_age_group(40) == "40-49"

    def test_age_group_boundary_50_is_50_plus(self):
        assert assign_age_group(50) == "50+"

    def test_prediction_dirty_string_inputs(self):
        """Whitespace and mixed-case strings must be sanitized correctly."""
        predictor = get_predictor()
        result = predictor.predict_single(
            " 40 ", " FEMALE ", " 28.0 ", " 1 ", " YES ", " NORTHEAST "
        )
        assert result["gender"] in ["Female", "Male"]
        assert result["smoking_status"] in ["Yes", "No"]
        assert result["region"] in ["Northeast", "Northwest", "Southeast", "Southwest"]
        assert result["predicted_annual_cost"] > 0

    def test_risk_gauge_score_exactly_0(self):
        fig = create_risk_gauge_chart(0.0, "LOW")
        assert isinstance(fig, go.Figure)

    def test_risk_gauge_score_exactly_100(self):
        fig = create_risk_gauge_chart(100.0, "HIGH")
        assert isinstance(fig, go.Figure)

    def test_risk_gauge_unknown_category(self):
        """Unknown category string must not raise — defaults to fallback colour."""
        fig = create_risk_gauge_chart(50.0, "UNKNOWN")
        assert isinstance(fig, go.Figure)

    def test_all_four_bmi_categories_in_processed(self):
        df = _load_proc()
        cats = set(df["bmi_category"].unique())
        for cat in ["Underweight", "Normal", "Overweight", "Obese"]:
            assert cat in cats, f"BMI category '{cat}' missing from processed dataset"

    def test_all_four_age_groups_in_processed(self):
        df = _load_proc()
        groups = set(df["age_group"].unique())
        for g in ["18-29", "30-39", "40-49", "50+"]:
            assert g in groups, f"Age group '{g}' missing from processed dataset"

    def test_processed_dataset_has_both_smoker_values(self):
        df = _load_proc()
        assert "yes" in df["smoker"].unique()
        assert "no"  in df["smoker"].unique()

    def test_scenario_comparison_single_row(self):
        """Chart must work with a single-entry scenario DataFrame."""
        df_sc = pd.DataFrame([{"Scenario": "Only A", "Predicted Annual Cost": 12000.0}])
        fig = create_scenario_comparison_chart(df_sc)
        assert isinstance(fig, go.Figure)
