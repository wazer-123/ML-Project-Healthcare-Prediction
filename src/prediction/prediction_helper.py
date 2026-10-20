"""
Prediction Helper Module
Handles model loading, feature validation, inference generation,
risk assessment, and non-causal explanation generation.
"""

import os
import traceback
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
    CATEGORICAL_COLS,
)


class InsurancePredictor:
    """Encapsulates model artifacts and inference logic."""

    def __init__(self, model_dir: str | None = None):
        if model_dir is None:
            # Resolve the project root from this file.
            project_root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )

            model_dir = os.path.join(project_root, "models")

        self.model_dir = os.path.abspath(model_dir)

        self.model_path = os.path.join(
            self.model_dir, "best_model.pkl"
        )

        self.preprocessor_path = os.path.join(
            self.model_dir, "preprocessing_pipeline.pkl"
        )

        self.all_models_path = os.path.join(
            self.model_dir, "all_models.pkl"
        )

        self.model = None
        self.preprocessor = None
        self.all_models = {}

        self.is_loaded = False
        self.load_error = None

        self.load_artifacts()

    # --------------------------------------------------
    # MODEL LOADING
    # --------------------------------------------------
    def load_artifacts(self) -> bool:
        """Load model artifacts and preserve detailed error information."""

        self.is_loaded = False
        self.load_error = None

        self.model = None
        self.preprocessor = None
        self.all_models = {}

        # Check model directory.
        if not os.path.isdir(self.model_dir):
            self.load_error = (
                f"Models directory does not exist: {self.model_dir}"
            )
            print(f"[MODEL ERROR] {self.load_error}")
            return False

        # Check required model.
        if not os.path.isfile(self.model_path):
            self.load_error = (
                f"Trained model file not found: {self.model_path}"
            )
            print(f"[MODEL ERROR] {self.load_error}")
            return False

        # Check required preprocessing pipeline.
        if not os.path.isfile(self.preprocessor_path):
            self.load_error = (
                "Preprocessing pipeline not found: "
                f"{self.preprocessor_path}"
            )
            print(f"[MODEL ERROR] {self.load_error}")
            return False

        try:
            print(f"[MODEL] Loading model: {self.model_path}")

            self.model = joblib.load(self.model_path)

            print(
                f"[MODEL] Loaded model type: "
                f"{type(self.model).__name__}"
            )

        except Exception as exc:
            self.load_error = (
                f"Failed to load best_model.pkl: "
                f"{type(exc).__name__}: {exc}"
            )

            print(f"[MODEL ERROR] {self.load_error}")
            traceback.print_exc()

            self.model = None
            return False

        try:
            print(
                "[MODEL] Loading preprocessing pipeline: "
                f"{self.preprocessor_path}"
            )

            self.preprocessor = joblib.load(
                self.preprocessor_path
            )

            print(
                f"[MODEL] Loaded preprocessor type: "
                f"{type(self.preprocessor).__name__}"
            )

        except Exception as exc:
            self.load_error = (
                "Failed to load preprocessing_pipeline.pkl: "
                f"{type(exc).__name__}: {exc}"
            )

            print(f"[MODEL ERROR] {self.load_error}")
            traceback.print_exc()

            self.model = None
            self.preprocessor = None
            return False

        # Additional models are optional. Their failure should not
        # prevent predictions using best_model.pkl.
        if os.path.isfile(self.all_models_path):

            try:
                print(
                    "[MODEL] Loading optional models: "
                    f"{self.all_models_path}"
                )

                self.all_models = joblib.load(
                    self.all_models_path
                )

                if not isinstance(self.all_models, dict):
                    raise TypeError(
                        "all_models.pkl must contain a dictionary."
                    )

                print(
                    f"[MODEL] Loaded {len(self.all_models)} "
                    "additional models."
                )

            except Exception as exc:
                # Record the warning, but retain the working main model.
                warning = (
                    f"Optional all_models.pkl could not be loaded: "
                    f"{type(exc).__name__}: {exc}"
                )

                print(f"[MODEL WARNING] {warning}")

                self.all_models = {}

        self.is_loaded = True
        self.load_error = None

        print("[MODEL] Required artifacts loaded successfully.")

        return True

    # --------------------------------------------------
    # SINGLE CUSTOMER PREDICTION
    # --------------------------------------------------
    def predict_single(
        self,
        age: int | float | str,
        sex: float | int | str,
        bmi: float | int | str,
        children: int | float | str,
        smoker: float | int | str,
        region: float | int | str,
        model_name: str | None = None,
    ) -> Dict[str, Any]:
        """Generate a prediction and a heuristic risk profile."""

        if (
            not self.is_loaded
            or self.preprocessor is None
            or self.model is None
        ):
            if not self.load_artifacts():
                raise RuntimeError(
                    "Model artifacts could not be loaded. "
                    f"Details: {self.load_error}"
                )

        if self.preprocessor is None or self.model is None:
            raise RuntimeError(
                "Artifacts loaded, but the model or preprocessor "
                "is unavailable."
            )

        # ----------------------------------------------
        # Validate age
        # ----------------------------------------------
        try:
            age_value = float(age)

            if not np.isfinite(age_value):
                age_value = 35.0

        except (TypeError, ValueError):
            age_value = 35.0

        age_clean = max(18, min(100, round(age_value)))

        # ----------------------------------------------
        # Validate BMI
        # ----------------------------------------------
        try:
            bmi_value = float(bmi)

            if not np.isfinite(bmi_value):
                bmi_value = 28.0

        except (TypeError, ValueError):
            bmi_value = 28.0

        bmi_clean = max(10.0, min(80.0, bmi_value))

        # ----------------------------------------------
        # Validate children
        # ----------------------------------------------
        try:
            children_value = float(children)

            if not np.isfinite(children_value):
                children_value = 0.0

        except (TypeError, ValueError):
            children_value = 0.0

        children_clean = max(
            0,
            min(10, round(children_value))
        )

        # ----------------------------------------------
        # Normalize categorical values
        # ----------------------------------------------
        sex_value = str(sex or "").lower().strip()

        sex_clean = (
            "female"
            if sex_value in ["female", "f", "woman"]
            else "male"
        )

        smoker_value = str(smoker or "").lower().strip()

        smoker_clean = (
            "yes"
            if smoker_value in [
                "yes",
                "y",
                "true",
                "1",
                "smoker",
            ]
            else "no"
        )

        region_value = (
            str(region or "")
            .lower()
            .strip()
            .replace(" ", "")
            .replace("-", "")
        )

        valid_regions = {
            "northeast": "northeast",
            "northwest": "northwest",
            "southeast": "southeast",
            "southwest": "southwest",
        }

        region_clean = valid_regions.get(
            region_value,
            "southwest",
        )

        # ----------------------------------------------
        # Prepare model input
        # ----------------------------------------------
        feature_order = NUMERICAL_COLS + CATEGORICAL_COLS

        input_values = {
            "age": age_clean,
            "bmi": bmi_clean,
            "children": children_clean,
            "sex": sex_clean,
            "smoker": smoker_clean,
            "region": region_clean,
        }

        missing_features = [
            feature
            for feature in feature_order
            if feature not in input_values
        ]

        if missing_features:
            raise ValueError(
                "Required input features are missing: "
                f"{missing_features}"
            )

        input_data = pd.DataFrame(
            [input_values]
        )[feature_order]

        # ----------------------------------------------
        # Transform input
        # ----------------------------------------------
        try:
            transformed_data = self.preprocessor.transform(
                input_data
            )

        except Exception as exc:
            raise RuntimeError(
                "Failed to transform customer input using the "
                f"preprocessing pipeline: {type(exc).__name__}: {exc}"
            ) from exc

        # ----------------------------------------------
        # Select prediction model
        # ----------------------------------------------
        selected_model = self.model

        if model_name and model_name in self.all_models:
            selected_model = self.all_models[model_name]

        # ----------------------------------------------
        # Generate prediction
        # ----------------------------------------------
        try:
            prediction = selected_model.predict(
                transformed_data
            )

            predicted_annual = float(prediction[0])

            if not np.isfinite(predicted_annual):
                raise ValueError(
                    "The model returned a non-finite prediction."
                )

            predicted_annual = max(
                800.0,
                round(predicted_annual, 2),
            )

            predicted_monthly = round(
                predicted_annual / 12.0,
                2,
            )

        except Exception as exc:
            raise RuntimeError(
                "Failed to generate insurance cost prediction: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # ----------------------------------------------
        # Calculate heuristic risk profile
        # ----------------------------------------------
        risk_score = calculate_heuristic_risk_score(
            age_clean,
            bmi_clean,
            smoker_clean,
            children_clean,
        )

        risk_category = assign_risk_category(
            risk_score
        )

        bmi_category = assign_bmi_category(
            bmi_clean
        )

        age_group = assign_age_group(
            age_clean
        )

        explanation_insights = self._generate_explanation(
            age_clean,
            bmi_clean,
            smoker_clean,
            children_clean,
            region_clean,
        )

        # ----------------------------------------------
        # Return prediction result
        # ----------------------------------------------
        return {
            "age": age_clean,
            "bmi": round(bmi_clean, 2),
            "predicted_annual_cost": predicted_annual,
            "predicted_monthly_cost": predicted_monthly,
            "risk_score": risk_score,
            "risk_category": risk_category,
            "age_category": age_group,
            "bmi_category": bmi_category,
            "smoking_status": smoker_clean.capitalize(),
            "dependents": children_clean,
            "region": region_clean.capitalize(),
            "gender": sex_clean.capitalize(),
            "sex": sex_clean.capitalize(),
            "smoker": smoker_clean.capitalize(),
            "children": children_clean,
            "explanation_insights": explanation_insights,
        }

    # --------------------------------------------------
    # EXPLANATION GENERATION
    # --------------------------------------------------
    def _generate_explanation(
        self,
        age: int,
        bmi: float,
        smoker: str,
        children: int,
        region: str,
    ) -> list:
        """Generate non-causal explanations of model associations."""

        insights = []

        # Smoking association
        if smoker.lower() == "yes":

            insights.append({
                "feature": "Smoking Status (Smoker)",
                "impact": "Strong Upward Association",
                "details": (
                    "In the predictive model, smoking status is "
                    "associated with higher predicted insurance costs. "
                    "This is a statistical association, not a causal "
                    "guarantee for an individual."
                ),
            })

        else:

            insights.append({
                "feature": "Smoking Status (Non-smoker)",
                "impact": "Favorable Association",
                "details": (
                    "Non-smoking profiles are associated with the "
                    "baseline cost tier in this predictive model."
                ),
            })

        # BMI association
        if bmi >= 30.0:

            compound = (
                " A combined association with smoking may also "
                "appear in the model."
                if smoker.lower() == "yes"
                else ""
            )

            insights.append({
                "feature": f"BMI ({bmi} - Obese Category)",
                "impact": "Moderate-to-High Upward Association",
                "details": (
                    f"A BMI of {bmi} falls in the obese category "
                    "and is associated with higher model predictions."
                    f"{compound}"
                ),
            })

        elif bmi >= 25.0:

            insights.append({
                "feature": f"BMI ({bmi} - Overweight Category)",
                "impact": "Slight Upward Association",
                "details": (
                    f"A BMI of {bmi} falls in the overweight category "
                    "and may be associated with higher model predictions."
                ),
            })

        else:

            insights.append({
                "feature": f"BMI ({bmi} - Normal/Optimal Range)",
                "impact": "Neutral/Favorable Association",
                "details": (
                    f"A BMI of {bmi} is below the overweight threshold. "
                    "This does not guarantee a lower insurance cost."
                ),
            })

        # Age association
        if age >= 50:

            insights.append({
                "feature": f"Age ({age} years)",
                "impact": "Moderate Upward Association",
                "details": (
                    "Older age groups may have higher predicted "
                    "insurance costs in this dataset."
                ),
            })

        elif age <= 30:

            insights.append({
                "feature": f"Age ({age} years)",
                "impact": "Favorable Association",
                "details": (
                    "Younger age groups are associated with lower "
                    "predicted costs in this dataset."
                ),
            })

        else:

            insights.append({
                "feature": f"Age ({age} years)",
                "impact": "Moderate Trend Progression",
                "details": (
                    "This age group may show a different cost pattern "
                    "from younger and older groups."
                ),
            })

        # Dependents
        if children > 0:

            insights.append({
                "feature": f"Dependents ({children} Children)",
                "impact": "Incremental Association",
                "details": (
                    f"The customer has {children} dependent(s). "
                    "The model uses this information when estimating cost."
                ),
            })

        else:

            insights.append({
                "feature": "Dependents (0 Children)",
                "impact": "No Dependents",
                "details": (
                    "The model estimates cost for a customer "
                    "with no dependent children."
                ),
            })

        # Region
        insights.append({
            "feature": f"Region ({region.capitalize()})",
            "impact": "Geographic Association",
            "details": (
                f"The selected region is {region.capitalize()}. "
                "Regional patterns in the training dataset may affect "
                "the model prediction."
            ),
        })

        return insights


# --------------------------------------------------
# CACHED PREDICTOR INSTANCE
# --------------------------------------------------
_predictor_instance = None


def get_predictor() -> InsurancePredictor:
    """Return the cached predictor instance."""

    global _predictor_instance

    if _predictor_instance is None:
        _predictor_instance = InsurancePredictor()

    return _predictor_instance