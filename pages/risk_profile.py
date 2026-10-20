"""
Customer Risk Profile Page
Generates a detailed actuarial profile for an individual applicant,
documents categorical health classifications, renders calibrated risk gauge visualizers,
and details the documented risk threshold logic.
"""

import os
import sys
import streamlit as st

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction.prediction_helper import get_predictor
from src.analytics.analytics_helper import create_risk_gauge_chart

def render_risk_profile():
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-badge">Actuarial Assessment</span>
        <div class="hero-title">Customer Risk Profile</div>
        <div class="hero-subtitle">Comprehensive Holistic Health & Demographic Evaluation</div>
    </div>
    """, unsafe_allow_html=True)

    predictor = get_predictor()
    if not predictor.is_loaded:
        st.error("⚠️ Trained model artifact not found. Please train models first.")
        st.stop()

    # Retrieve from session state or load standard benchmark profile
    if 'last_prediction' in st.session_state:
        profile = st.session_state['last_prediction']
        st.success("✅ Displaying the most recent applicant profile generated in the Insurance Cost Predictor.")
    else:
        st.info("ℹ️ No active applicant session detected. Displaying a standard demonstration profile. You can modify parameters below.")
        profile = predictor.predict_single(
            age=45,
            sex='male',
            bmi=29.8,
            children=2,
            smoker='no',
            region='southwest'
        )

    # Profile Parameter Adjustment Expander
    with st.expander("⚙️ Adjust Profile Parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            adj_age = st.slider("Age", 18, 80, 45)
            adj_gender = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            adj_bmi = st.slider("BMI", 15.0, 50.0, 29.8, step=0.1)
            adj_children = st.selectbox("Children", [0, 1, 2, 3, 4, 5], index=2)
        with c3:
            adj_smoker = st.selectbox("Smoker", ["No", "Yes"], index=0)
            adj_region = st.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"])

        if st.button("🔄 Update Profile & Recalculate"):
            profile = predictor.predict_single(
                age=adj_age,
                sex=adj_gender.lower(),
                bmi=adj_bmi,
                children=adj_children,
                smoker=adj_smoker.lower(),
                region=adj_region.lower()
            )
            st.session_state['last_prediction'] = profile
            st.rerun()

    # Primary Profile Details Grid
    st.markdown("### 👤 Applicant Demographic & Risk Breakdown")

    col_details, col_gauge = st.columns([1.2, 1])

    with col_details:
        st.markdown(f"""
        <div class="glass-panel">
            <h4 style="color:#0f172a; margin-bottom:16px;">Applicant Profile Attributes</h4>
            <table style="width:100%; font-size:0.95rem; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #f1f5f9; padding: 8px 0;">
                    <td style="padding: 10px 0; color:#64748b;"><b>Age & Demographic Tier</b></td>
                    <td style="padding: 10px 0; font-weight:600; text-align:right;">{profile.get('age', 'N/A')} yrs ({profile['age_category']})</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 0; color:#64748b;"><b>BMI & Classification</b></td>
                    <td style="padding: 10px 0; font-weight:600; text-align:right;">{profile.get('bmi', 'N/A')} ({profile['bmi_category']})</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 0; color:#64748b;"><b>Smoking Status</b></td>
                    <td style="padding: 10px 0; font-weight:600; text-align:right;">{profile['smoking_status']}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 0; color:#64748b;"><b>Dependents (Children)</b></td>
                    <td style="padding: 10px 0; font-weight:600; text-align:right;">{profile['dependents']}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 0; color:#64748b;"><b>Geographical Region</b></td>
                    <td style="padding: 10px 0; font-weight:600; text-align:right;">{profile['region']}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 0; color:#64748b;"><b>Projected Annual Premium</b></td>
                    <td style="padding: 10px 0; font-weight:700; color:#0284c7; text-align:right;">${profile['predicted_annual_cost']:,.2f}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 0; color:#64748b;"><b>Composite Risk Score</b></td>
                    <td style="padding: 10px 0; font-weight:700; text-align:right;">{profile['risk_score']} / 100</td>
                </tr>
                <tr>
                    <td style="padding: 10px 0; color:#64748b;"><b>Risk Classification Tier</b></td>
                    <td style="padding: 10px 0; text-align:right;"><span class="risk-badge-{profile['risk_category'].lower()}">{profile['risk_category']}</span></td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        st.markdown("#### Actuarial Risk Index Gauge")
        fig_gauge = create_risk_gauge_chart(profile['risk_score'], profile['risk_category'])
        st.plotly_chart(fig_gauge, width='stretch')

    # Documented Threshold Logic
    st.markdown("### 📖 Documented Risk Threshold Logic")
    st.markdown("""
    The composite risk score is evaluated on a standardized scale from **0.0 to 100.0** utilizing actuarial criteria:

    | Risk Tier | Score Band | Demographic & Health Profile Criteria | Typical Actuarial Implication |
    | :--- | :--- | :--- | :--- |
    | **LOW** | **0.0 – 34.9** | Non-smoker, BMI < 28 (Normal to mild overweight), Age < 45, no compound risk indicators. | Standard baseline claim frequency; standard premium bracket. |
    | **MEDIUM** | **35.0 – 69.9** | Non-smoker with elevated BMI (28-35), senior age bracket (50+), or moderate dependent pooling. | Moderate likelihood of chronic condition management. |
    | **HIGH** | **70.0 – 100.0** | Active smoker, clinical obesity (BMI ≥ 30), or compound smoker-obesity interaction. | Significant statistical probability of high-dollar clinical outlays. |
    """)

    st.markdown("""
    <div class="disclaimer-card">
        <b>Mandatory Regulatory Disclaimer:</b> This risk profile is a <b>model-based synthetic indicator</b> designed strictly for analytical exploration, scenario simulation, and educational demonstration. It is <b>NOT</b> an actual insurance underwriting decision, clinical diagnostic assessment, or binding policy rating.
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__' or (hasattr(st, 'runtime') and st.runtime.exists()):
    render_risk_profile()
