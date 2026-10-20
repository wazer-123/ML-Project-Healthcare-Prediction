import os
import sys
import streamlit as st
import pandas as pd

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# PROJECT PATH CONFIGURATION
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# --------------------------------------------------
# IMPORT PREDICTION MODULE
# --------------------------------------------------
try:
    from src.prediction.prediction_helper import get_predictor

    predictor_import_error = None

except Exception as e:
    predictor_import_error = str(e)
    get_predictor = None


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------
st.title("🏥 Insurance Cost Predictor")

st.write(
    "Estimate annual and monthly health insurance costs "
    "using a trained machine learning model."
)

st.divider()


# --------------------------------------------------
# DEPLOYMENT DIAGNOSTICS
# --------------------------------------------------
with st.expander("🔍 Deployment Diagnostics", expanded=True):

    st.write("**Project root:**", PROJECT_ROOT)

    st.write(
        "**Models directory exists:**",
        os.path.isdir(MODELS_DIR)
    )

    if os.path.isdir(MODELS_DIR):
        model_files = os.listdir(MODELS_DIR)

        st.write("**Available model files:**")

        if model_files:
            for filename in model_files:
                file_path = os.path.join(MODELS_DIR, filename)

                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)

                    st.write(
                        f"- `{filename}` — "
                        f"{file_size:,} bytes"
                    )
        else:
            st.warning("The models directory is empty.")

    else:
        st.error("The models directory was not found.")

    if predictor_import_error:
        st.error(
            "Prediction module import failed: "
            f"{predictor_import_error}"
        )


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------
predictor = None
model_loading_error = None

if get_predictor is not None:

    try:
        predictor = get_predictor()

        with st.expander("🧪 Model Loading Status", expanded=True):

            st.write(
                "**Predictor loaded:**",
                getattr(predictor, "is_loaded", False)
            )

            st.write(
                "**Model path:**",
                getattr(predictor, "model_path", "Not available")
            )

            st.write(
                "**Preprocessor path:**",
                getattr(
                    predictor,
                    "preprocessor_path",
                    "Not available"
                )
            )

            model_path = getattr(predictor, "model_path", None)
            preprocessor_path = getattr(
                predictor,
                "preprocessor_path",
                None
            )

            if model_path:
                st.write(
                    "**Model file exists:**",
                    os.path.isfile(model_path)
                )

            if preprocessor_path:
                st.write(
                    "**Preprocessor file exists:**",
                    os.path.isfile(preprocessor_path)
                )

    except Exception as e:
        model_loading_error = str(e)

        st.error(
            f"An error occurred while initializing the model: "
            f"{model_loading_error}"
        )


# --------------------------------------------------
# STOP IF MODEL IS NOT AVAILABLE
# --------------------------------------------------
if predictor is None:

    st.error(
        "The prediction module could not be initialized. "
        "Check the Deployment Diagnostics above."
    )

    st.stop()


if not getattr(predictor, "is_loaded", False):

    st.error(
        "The trained ML model could not be loaded. "
        "Check the model paths and deployment diagnostics."
    )

    st.info(
        "The diagnostic output will help us determine whether "
        "the problem is a missing artifact, an incorrect path, "
        "or an error while loading the model."
    )

    st.stop()


# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.subheader("📝 Enter Customer Information")

with st.form("insurance_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

        sex = st.selectbox(
            "Gender",
            options=["male", "female"]
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=80.0,
            value=26.0,
            step=0.1
        )

    with col2:

        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

        smoker = st.selectbox(
            "Smoking Status",
            options=["no", "yes"]
        )

        region = st.selectbox(
            "Region",
            options=[
                "northeast",
                "northwest",
                "southeast",
                "southwest"
            ]
        )

    submitted = st.form_submit_button(
        "🔮 Predict Insurance Cost",
        use_container_width=True
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if submitted:

    try:

        with st.spinner("Calculating insurance cost..."):

            result = predictor.predict_single(
                age=age,
                sex=sex,
                bmi=bmi,
                children=children,
                smoker=smoker,
                region=region
            )

        if not isinstance(result, dict):

            st.error(
                "The prediction function returned an unexpected result."
            )

            st.write("Returned result:", result)

            st.stop()

        # ------------------------------------------
        # DISPLAY PREDICTION RESULTS
        # ------------------------------------------
        st.divider()

        st.subheader("📊 Prediction Results")

        annual_cost = result.get("predicted_annual_cost")
        monthly_cost = result.get("predicted_monthly_cost")

        col1, col2 = st.columns(2)

        with col1:

            if annual_cost is not None:
                st.metric(
                    label="Estimated Annual Insurance Cost",
                    value=f"${float(annual_cost):,.2f}"
                )
            else:
                st.warning(
                    "Annual cost was not returned by the model."
                )

        with col2:

            if monthly_cost is not None:
                st.metric(
                    label="Estimated Monthly Insurance Cost",
                    value=f"${float(monthly_cost):,.2f}"
                )
            else:
                st.warning(
                    "Monthly cost was not returned by the model."
                )

        # ------------------------------------------
        # RISK INFORMATION
        # ------------------------------------------
        st.subheader("⚠️ Customer Risk Profile")

        risk_col1, risk_col2 = st.columns(2)

        with risk_col1:

            risk_score = result.get("risk_score")

            if risk_score is not None:
                st.metric(
                    label="Risk Score",
                    value=f"{float(risk_score):.1f}"
                )

        with risk_col2:

            risk_category = result.get("risk_category")

            if risk_category is not None:
                st.metric(
                    label="Risk Category",
                    value=str(risk_category)
                )

        # ------------------------------------------
        # ADDITIONAL DETAILS
        # ------------------------------------------
        st.subheader("👤 Customer Details")

        details = {
            "Age": result.get("age", age),
            "Gender": result.get("sex", sex),
            "BMI": result.get("bmi", bmi),
            "Children": result.get("children", children),
            "Smoker": result.get("smoker", smoker),
            "Region": result.get("region", region),
            "Age Category": result.get("age_category", "N/A"),
            "BMI Category": result.get("bmi_category", "N/A")
        }

        details_df = pd.DataFrame(
            list(details.items()),
            columns=["Feature", "Value"]
        )

        st.dataframe(
            details_df,
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------
        # COMPLETE RESULT
        # ------------------------------------------
        with st.expander("View Complete Prediction Output"):

            st.json(result)

        st.caption(
            "Note: This prediction is an estimate from a machine "
            "learning model. Actual insurance costs may differ."
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)

        st.info(
            "Please share the complete error message so we can "
            "identify and fix the underlying issue."
        )