"""
Insurance Cost Predictor Page
Provides an intuitive, professional interface for actuarial cost estimation,
risk indexing, non-causal statistical associations, and result export.
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
    # Hero Title Banner
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-badge">AI Predictive Engine</span>
        <div class="hero-title">Insurance Cost Predictor</div>
        <div class="hero-subtitle">Real-Time Machine Learning Inference & Actuarial Risk Profiling</div>
    </div>
    """, unsafe_allow_html=True)

    predictor = get_predictor()
    if not getattr(predictor, "is_loaded", False):
        st.error(
            "⚠️ Trained machine learning model artifacts could not be loaded. "
            "Please ensure the model training pipeline has completed successfully."
        )
        st.stop()

    st.markdown("### 📋 Enter Applicant Information")
    st.caption("Provide customer demographics and lifestyle indicators to generate premium estimates.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age (Years)",
                min_value=18,
                max_value=100,
                value=35,
                step=1,
                help="Chronological age of primary applicant (18 - 100)"
            )
            gender = st.selectbox(
                "Gender",
                options=["Male", "Female"],
                index=0,
                help="Biological sex recorded for actuarial rating"
            )

        with col2:
            bmi = st.number_input(
                "BMI (Body Mass Index)",
                min_value=10.0,
                max_value=80.0,
                value=27.5,
                step=0.1,
                format="%.1f",
                help="Weight (kg) / [Height (m)]² — WHO standard scale"
            )
            children = st.selectbox(
                "Number of Dependents / Children",
                options=[0, 1, 2, 3, 4, 5],
                index=1,
                help="Count of dependent children covered under policy"
            )

        with col3:
            smoker = st.selectbox(
                "Smoking Status",
                options=["No", "Yes"],
                index=0,
                help="Active tobacco or nicotine consumption indicator"
            )
            region = st.selectbox(
                "Geographic Region",
                options=["Northeast", "Northwest", "Southeast", "Southwest"],
                index=0,
                help="Residential rating jurisdiction within the United States"
            )

        submitted = st.form_submit_button("⚡ PREDICT INSURANCE COST", use_container_width=True)

    # Process prediction if submitted
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
                st.error(f"Inference error: {e}")
                st.stop()

    # Display results when prediction is available
    if 'last_prediction' in st.session_state:
        res = st.session_state['last_prediction']

        st.markdown("---")
        st.markdown("### 🎯 Predictive Assessment Summary")

        # Key output metrics (4 KPI cards)
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
                <div class="kpi-sub">Prorated Over 12 Months</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #f59e0b;">
                <div class="kpi-title">Actuarial Risk Score</div>
                <div class="kpi-value">{res['risk_score']} <span style="font-size:1rem;color:#64748b;">/ 100</span></div>
                <div class="kpi-sub">Heuristic Risk Index</div>
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
            st.plotly_chart(fig_gauge, use_container_width=True)

        with exp_col:
            st.markdown("#### 💡 Prediction Insights")
            st.caption("Statistical factors associated with the model prediction (non-causal associations):")

            for item in res.get('explanation_insights', []):
                with st.expander(f"📌 {item['feature']} — {item['impact']}", expanded=True):
                    st.write(item['details'])

        # Detailed breakdown table
        st.markdown("#### 👤 Applicant Profile Breakdown")
        profile_data = {
            "Attribute": [
                "Age", "Gender", "BMI", "BMI Category", "Age Group",
                "Dependents", "Smoking Status", "Region"
            ],
            "Value": [
                f"{res.get('age', age)} years",
                str(res.get('gender', gender)).capitalize(),
                f"{res.get('bmi', bmi):.1f}",
                str(res.get('bmi_category', 'N/A')),
                str(res.get('age_category', 'N/A')),
                f"{res.get('dependents', children)}",
                str(res.get('smoking_status', smoker)).capitalize(),
                str(res.get('region', region)).capitalize()
            ]
        }
        st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True)

        # Download prediction report
        st.markdown("#### 📥 Export Prediction Record")
        pred_record = pd.DataFrame([{
            'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Age': res.get('age', age),
            'Gender': res.get('gender', gender),
            'BMI': res.get('bmi', bmi),
            'Age_Category': res.get('age_category', 'N/A'),
            'BMI_Category': res.get('bmi_category', 'N/A'),
            'Dependents': res.get('dependents', children),
            'Smoking_Status': res.get('smoking_status', smoker),
            'Region': res.get('region', region),
            'Predicted_Annual_Cost': res['predicted_annual_cost'],
            'Predicted_Monthly_Cost': res['predicted_monthly_cost'],
            'Risk_Score': res['risk_score'],
            'Risk_Category': res['risk_category']
        }])

        csv_data = pred_record.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Prediction Result (CSV)",
            data=csv_data,
            file_name=f"insurance_prediction_{res['risk_category'].lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("""
        <div class="disclaimer-card">
            <b>Underwriting Advisory Disclaimer:</b> This projection is produced by a trained machine learning regression model for estimation and scenario evaluation purposes. It is based on empirical patterns in historical data and does not constitute formal underwriting acceptance, binding insurance quotation, or clinical medical diagnosis.
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
                <b style="color: #0284c7;">"⚡ PREDICT INSURANCE COST"</b> to compute machine learning cost projections,
                actuarial risk tiering, and statistical associations.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__' or (hasattr(st, 'runtime') and st.runtime.exists()):
    render_predictor()