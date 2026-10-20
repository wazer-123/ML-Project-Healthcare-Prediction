"""
Insurance Cost Predictor Page
Provides interactive user inputs for demographic and health factors,
queries the trained ML model for accurate inference, displays risk scores,
and presents non-causal statistical explanation insights.
"""

import os
import sys
import streamlit as st
import pandas as pd

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction.prediction_helper import get_predictor
from src.analytics.analytics_helper import create_risk_gauge_chart

def render_predictor():
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-badge">AI Predictive Engine</span>
        <div class="hero-title">Insurance Cost Predictor</div>
        <div class="hero-subtitle">Real-Time ML Inference & Risk Profiling</div>
    </div>
    """, unsafe_allow_html=True)

    predictor = get_predictor()
    if not predictor.is_loaded:
        st.error("⚠️ Trained ML model artifact not found. Please execute the training pipeline first.")
        st.stop()

    st.markdown("### 📋 Enter Applicant Information")
    st.caption("Provide customer demographics and lifestyle indicators to generate premium estimates.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age (Years)", min_value=18, max_value=85, value=35, step=1, help="Age of the primary applicant (18-85)")
            gender = st.selectbox("Gender", options=["Male", "Female"], index=0)

        with col2:
            bmi = st.number_input("BMI (Body Mass Index)", min_value=14.0, max_value=55.0, value=27.5, step=0.1, help="Weight (kg) / [Height (m)]^2")
            children = st.selectbox("Number of Dependents / Children", options=[0, 1, 2, 3, 4, 5], index=1)

        with col3:
            smoker = st.selectbox("Smoking Status", options=["No", "Yes"], index=0, help="Does the individual smoke tobacco or nicotine products?")
            region = st.selectbox("Geographic Region", options=["Northeast", "Northwest", "Southeast", "Southwest"], index=0)

        submitted = st.form_submit_button("⚡ PREDICT INSURANCE COST", width='stretch')

    # Process prediction if submitted or retrieve cached
    if submitted:
        with st.spinner("Processing actuarial machine learning inference..."):
            try:
                result = predictor.predict_single(
                    age=age,
                    sex=gender.lower(),
                    bmi=bmi,
                    children=children,
                    smoker=smoker.lower(),
                    region=region.lower()
                )
                st.session_state['last_prediction'] = result
            except Exception as e:
                st.error(f"Prediction computation error: {e}")
                st.stop()

    # Display results
    if 'last_prediction' in st.session_state:
        res = st.session_state['last_prediction']

        st.markdown("---")
        st.markdown("### 🎯 Predictive Assessment Summary")

        # Key output metrics
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #0284c7;">
                <div class="kpi-title">Predicted Annual Cost</div>
                <div class="kpi-value" style="color: #0284c7;">${res['predicted_annual_cost']:,.2f}</div>
                <div class="kpi-sub">Estimated Total Annual Charge</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #6366f1;">
                <div class="kpi-title">Estimated Monthly Cost</div>
                <div class="kpi-value" style="color: #6366f1;">${res['predicted_monthly_cost']:,.2f}</div>
                <div class="kpi-sub">Prorated / 12 Months</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #f59e0b;">
                <div class="kpi-title">Model Risk Score</div>
                <div class="kpi-value">{res['risk_score']} <span style="font-size:1rem;color:#64748b;">/ 100</span></div>
                <div class="kpi-sub">Actuarial Risk Index</div>
            </div>
            """, unsafe_allow_html=True)

        badge_class = f"risk-badge-{res['risk_category'].lower()}"
        with m4:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #10b981;">
                <div class="kpi-title">Risk Tier</div>
                <div class="kpi-value" style="font-size:1.4rem; padding-top:4px;">
                    <span class="{badge_class}">{res['risk_category']} RISK</span>
                </div>
                <div class="kpi-sub">Category Classification</div>
            </div>
            """, unsafe_allow_html=True)

        # Risk Gauge and Explanations
        g_col, exp_col = st.columns([1, 1.3])

        with g_col:
            st.markdown("#### Actuarial Risk Gauge")
            fig_gauge = create_risk_gauge_chart(res['risk_score'], res['risk_category'])
            st.plotly_chart(fig_gauge, width='stretch')

        with exp_col:
            st.markdown("#### 💡 Prediction Insights")
            st.caption("Statistical factors associated with the model prediction (non-causal associations):")

            for item in res['explanation_insights']:
                with st.expander(f"📌 {item['feature']} — {item['impact']}", expanded=True):
                    st.write(item['details'])

        # Download prediction report
        st.markdown("#### 📥 Export Prediction Record")
        pred_record = pd.DataFrame([{
            'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Age': res.get('age', age),
            'Gender': res['gender'],
            'BMI': res.get('bmi', bmi),
            'Age_Category': res['age_category'],
            'BMI_Category': res['bmi_category'],
            'Children': res['dependents'],
            'Smoker': res['smoking_status'],
            'Region': res['region'],
            'Predicted_Annual_Cost': res['predicted_annual_cost'],
            'Predicted_Monthly_Cost': res['predicted_monthly_cost'],
            'Risk_Score': res['risk_score'],
            'Risk_Category': res['risk_category']
        }])

        csv_data = pred_record.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Prediction Result (CSV)",
            data=csv_data,
            file_name=f"insurance_prediction_{res['risk_category'].lower()}.csv",
            mime="text/csv"
        )

        st.markdown("""
        <div class="disclaimer-card">
            <b>Underwriting Advisory Disclaimer:</b> This projection is produced by a trained machine learning regression model for estimation and scenario evaluation purposes. It does not constitute formal underwriting acceptance, binding insurance quotation, or clinical medical diagnosis.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("---")
        st.markdown("""
        <div class="glass-panel" style="text-align: center; padding: 36px 20px;">
            <div style="font-size: 2.2rem; margin-bottom: 10px;">🔮</div>
            <h3 style="color: #0f172a; margin-bottom: 6px;">Applicant Evaluation Ready</h3>
            <p style="color: #64748b; max-width: 580px; margin: 0 auto; font-size: 0.95rem; line-height: 1.5;">
                Adjust applicant parameters in the form above and click 
                <b style="color: #0284c7;">"⚡ PREDICT INSURANCE COST"</b> to compute machine learning cost projections, actuarial risk tiering, and non-causal statistical associations.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__' or (hasattr(st, 'runtime') and st.runtime.exists()):
    render_predictor()
