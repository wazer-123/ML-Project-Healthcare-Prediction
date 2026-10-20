"""
Customer Risk Profile Page
Generates a detailed actuarial profile for an individual applicant,
documents categorical health classifications, renders calibrated risk gauge visualizers,
and details the documented risk threshold logic and limitations.
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
        <div class="hero-subtitle">Comprehensive Demographic & Lifestyle Health Risk Indexing</div>
    </div>
    """, unsafe_allow_html=True)

    predictor = get_predictor()
    if not getattr(predictor, "is_loaded", False):
        st.error("⚠️ Trained model artifact not found. Please verify the model artifacts.")
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
    with st.expander("⚙️ Adjust Profile Parameters & Recalculate", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            adj_age = st.slider("Age", 18, 100, int(profile.get('age', 45)))
            adj_gender = st.selectbox("Gender", ["Male", "Female"], index=0 if str(profile.get('gender', 'male')).lower() == 'male' else 1)
        with c2:
            adj_bmi = st.slider("BMI", 10.0, 80.0, float(profile.get('bmi', 29.8)), step=0.1)
            adj_children = st.selectbox("Children", [0, 1, 2, 3, 4, 5], index=min(int(profile.get('dependents', 2)), 5))
        with c3:
            adj_smoker = st.selectbox("Smoker", ["No", "Yes"], index=1 if str(profile.get('smoking_status', 'no')).lower() == 'yes' else 0)
            reg_opts = ["Southwest", "Southeast", "Northwest", "Northeast"]
            reg_idx = 0
            if profile.get('region', 'Southwest').capitalize() in reg_opts:
                reg_idx = reg_opts.index(profile.get('region', 'Southwest').capitalize())
            adj_region = st.selectbox("Region", reg_opts, index=reg_idx)

        if st.button("🔄 Update Profile & Recalculate", use_container_width=True):
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
                    <td style="padding: 10px 0; font-weight:600; text-align:right;">{profile.get('age', 'N/A')} yrs ({profile.get('age_category', 'N/A')})</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 0; color:#64748b;"><b>BMI & Classification</b></td>
                    <td style="padding: 10px 0; font-weight:600; text-align:right;">{profile.get('bmi', 'N/A')} ({profile.get('bmi_category', 'N/A')})</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 0; color:#64748b;"><b>Smoking Status</b></td>
                    <td style="padding: 10px 0; font-weight:600; text-align:right;">{profile.get('smoking_status', 'N/A')}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 0; color:#64748b;"><b>Dependents (Children)</b></td>
                    <td style="padding: 10px 0; font-weight:600; text-align:right;">{profile.get('dependents', 'N/A')}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 0; color:#64748b;"><b>Geographical Region</b></td>
                    <td style="padding: 10px 0; font-weight:600; text-align:right;">{profile.get('region', 'N/A')}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 0; color:#64748b;"><b>Projected Annual Cost</b></td>
                    <td style="padding: 10px 0; font-weight:700; color:#0284c7; text-align:right;">${profile['predicted_annual_cost']:,.2f}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 0; color:#64748b;"><b>Actuarial Risk Index</b></td>
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
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Documented Threshold Logic
    st.markdown("### 📖 Documented Risk Threshold Logic")
    st.markdown("""
    The composite risk score is evaluated on a heuristic scale from **0.0 to 100.0** utilizing actuarial indicators:

    | Risk Tier | Score Band | Demographic & Health Profile Criteria | Typical Actuarial Implication |
    | :--- | :--- | :--- | :--- |
    | **LOW** | **0.0 – 34.9** | Non-smoker, BMI < 28 (Normal to mild overweight), Age < 45, no compound risk indicators. | Baseline claim frequency; standard premium bracket. |
    | **MEDIUM** | **35.0 – 69.9** | Non-smoker with elevated BMI (28-35), senior age bracket (50+), or moderate dependent pooling. | Moderate statistical variance over baseline cost tier. |
    | **HIGH** | **70.0 – 100.0** | Active smoker, clinical obesity (BMI ≥ 30), or compound smoker-obesity interaction. | Significant statistical probability of high-dollar clinical claim outlay. |
    """)

    st.markdown("""
    <div class="disclaimer-card">
        <b>Heuristic Scope & Limitations:</b> The risk score provided on this page is a rule-based actuarial heuristic
        constructed from demographic risk proxies (smoking status, BMI, age, and dependent count).
        It is strictly designed for analytical cohort benchmarking and educational demonstration.
        It is <b>NOT</b> a clinically validated medical diagnosis, biological health metric, or formal binding underwriting decision.
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__' or (hasattr(st, 'runtime') and st.runtime.exists()):
    render_risk_profile()
