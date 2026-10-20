# AI-Powered Insurance Cost & Risk Analytics Dashboard

An enterprise-grade, multi-page machine learning and actuarial analytics web application built with **Python, Streamlit, Scikit-learn, and Plotly**. Designed to evaluate individual insurance cost projections, model cohort risks, execute what-if sensitivity analyses, and inspect predictive machine learning diagnostics.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement & Objectives](#-problem-statement--objectives)
3. [Key Architectural Features](#-key-architectural-features)
4. [Technology Stack](#-technology-stack)
5. [Dataset Information](#-dataset-information)
6. [Machine Learning Workflow](#-machine-learning-workflow)
7. [Model Comparison & Benchmark Results](#-model-comparison--benchmark-results)
8. [Project Structure](#-project-structure)
9. [Installation & Setup (Windows)](#-installation--setup-windows)
10. [Model Training Instructions](#-model-training-instructions)
11. [Running the Application](#-running-the-application)
12. [Risk Logic & Non-Causal Disclaimers](#-risk-logic--non-causal-disclaimers)
13. [Limitations & Future Enhancements](#-limitations--future-enhancements)

---

## 🔍 Project Overview

The **AI-Powered Insurance Cost & Risk Analytics Dashboard** bridges predictive machine learning and actuarial health economics. Instead of simple, isolated single-page prediction forms, this platform provides an executive suite featuring:
- Dynamic portfolio dashboards with multi-dimensional filtering
- Real-time ML inference with actuarial risk scoring
- Calibrated risk gauge visualizers with documented tier logic
- Deep demographic and lifestyle exploratory analytics
- Multi-model evaluation benchmarking (Linear Regression, Random Forest, Gradient Boosting)
- High-fidelity What-If scenario simulations and sensitivity tools
- Comprehensive dataset governance and custom interactive charting

---

## 🎯 Problem Statement & Objectives

### Problem Statement
Healthcare insurance underwriting and premium estimation require balanced assessment of biological, demographic, and lifestyle risks. Underwriters and risk analysts need interactive simulation tools to test marginal cost sensitivities, diagnose algorithmic behavior, and understand non-linear interactions (e.g., the compounding risk of high BMI combined with smoking).

### Objectives
1. **Actuarial Accuracy:** Train and benchmark multiple regression algorithms to achieve high prediction fidelity ($R^2 > 0.95$).
2. **Explainability & Governance:** Provide non-causal statistical associations and feature importances to avoid misleading causal assumptions.
3. **Interactive Simulation:** Empower analysts to create and compare multi-applicant scenarios in real-time.
4. **Data Transparency:** Offer complete data exploration, summary statistics, and CSV data export across all pages.

---

## 🚀 Key Architectural Features

### 1. Executive Dashboard
- **KPI Summary Cards:** Total Customers, Average Cost, Median Cost, High-Risk Percentage, Min/Max Costs, Average Age, and Average BMI.
- **Interactive Plotly Visualizations:** Cost distribution with KDE, cost by age group, cost by BMI category, smoker breakdown, gender analysis, regional comparison, and age vs cost scatter plots.
- **Multi-Dimensional Filters:** Age range, gender, BMI range, smoking status, region, and dependents filter with instant chart updates.
- **Filtered Export:** Download filtered portfolio subsets directly as CSV.

### 2. Insurance Cost Predictor
- Real-time inference powered by the winning serialized production model (`Gradient Boosting Regressor`).
- Outputs: Predicted Annual Cost, Prorated Monthly Cost, Model Risk Score (0-100), and Risk Classification Tier (`LOW`, `MEDIUM`, `HIGH`).
- **Prediction Insights:** Granular breakdown of feature factors *associated with the model prediction* without claiming causal guarantees.
- Individual applicant record export as CSV.

### 3. Customer Risk Profile
- Categorical demographic classifications: Age category, BMI classification, smoking status, dependents, region, and predicted premium.
- Calibrated Plotly Gauge Chart with colored threshold bands:
  - **LOW Risk (0.0 – 34.9):** Non-smokers with standard BMI and lower claim frequency profiles.
  - **MEDIUM Risk (35.0 – 69.9):** Elevated BMI or senior age brackets.
  - **HIGH Risk (70.0 – 100.0):** Smokers, clinical obesity (BMI ≥ 30), or compound risk interactions.
- Regulatory advisory disclaimers.

### 4. Cost & Demographic Analytics
- Continuous factor correlation: Age vs Cost (trendline stratified by smoking) and BMI vs Cost (obesity threshold breakdown).
- Demographic analysis: Cost by gender, region, age group, and dependent counts.
- Risk segmentation: Distribution of insurance charges by risk category.

### 5. Model Performance & Diagnostics
- Multi-model comparison table: MAE, MSE, RMSE, and $R^2$ across Linear Regression, Random Forest, and Gradient Boosting.
- Automated winning model selection based on highest test $R^2$ and lowest RMSE.
- Actual vs Predicted regression charts with 45-degree ideal line.
- Residual scatter plots ($y_{pred}$ vs $residuals$) and error distribution histograms.
- Global feature importance horizontal ranking.

### 6. What-If Scenario Analysis
- **Multi-Cohort Comparison:** Compare Scenario A (Young Healthy), Scenario B (Mid-Age Overweight Smoker), Scenario C (Senior Obese Smoker), and Custom Scenario D.
- Calculates exact dollar difference ($\pm\$$) and percentage difference ($\pm\%$) from baseline.
- **Real-Time Baseline Sensitivity Simulator:** Live sliders for Age, BMI, Smoking, Children, Region, and Gender with real-time marginal deviation readouts.
- Export scenario matrix as CSV.

### 7. Data Explorer
- Portfolio metadata audit: Total records, feature count, missing value validation, and memory usage.
- Schema inspection: Data types, null counts, unique values, and sample values.
- Statistical descriptions for continuous and categorical attributes.
- **Interactive Visual Query Builder:** Select any X-axis, Y-axis, Color grouping, and Chart Type (Scatter, Box, Histogram, Bar, Violin) on the fly.
- Export full dataset as CSV.

---

## 🛠️ Technology Stack

| Component | Library / Framework | Version |
| :--- | :--- | :--- |
| **Language** | Python | 3.13.1 |
| **Web Interface** | Streamlit | ≥ 1.36.0 |
| **Data Manipulation** | Pandas, NumPy | ≥ 2.0.0 |
| **Machine Learning** | Scikit-Learn | ≥ 1.3.0 |
| **Interactive Charts** | Plotly | ≥ 5.18.0 |
| **Model Persistence** | Joblib | ≥ 1.3.0 |

---

## 📊 Dataset Information

- **Dataset Provenance:** **Synthetic/Demo Dataset** (6,000 records).
- **Labeling:** Explicitly designated as synthetic actuarial demo data; not presented as proprietary real-company records.
- **Features Included:**
  1. `age`: Integer (18 – 65)
  2. `sex`: Categorical (`male`, `female`)
  3. `bmi`: Float (15.0 – 53.0)
  4. `children`: Integer (0 – 5)
  5. `smoker`: Categorical (`yes`, `no`)
  6. `region`: Categorical (`southwest`, `southeast`, `northwest`, `northeast`)
  7. `charges`: Continuous target variable ($1,120 – $65,000)

---

## 🧠 Machine Learning Workflow

```
Raw Dataset Generation (6,000 samples)
       ↓
Schema Validation & Cleaning
       ↓
Feature Engineering (BMI Category, Age Group, Risk Scoring)
       ↓
Train / Test Split (80% Train, 20% Test, Seed 42)
       ↓
Scikit-Learn Pipeline (StandardScaler for numerics, OneHotEncoder for categoricals)
       ↓
Multi-Model Training (Linear Regression, Random Forest, Gradient Boosting)
       ↓
Evaluation Diagnostics (MAE, MSE, RMSE, R²)
       ↓
Automated Best Model Selection
       ↓
Artifact Serialization (Joblib: best_model.pkl & preprocessing_pipeline.pkl)
       ↓
Interactive Streamlit Dashboard & Live Inference
```

---

## 🏆 Model Comparison & Benchmark Results

Evaluated on held-out 20% test partition (1,200 samples):

| Model Architecture | Test $R^2$ Score | RMSE ($) | MAE ($) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting Regressor** | **0.9921** | **$1,018.01** | **$796.47** | 🏆 **ACTIVE PRODUCTION** |
| **Random Forest Regressor** | 0.9906 | $1,109.44 | $881.89 | Candidate |
| **Linear Regression** | 0.8725 | $4,091.43 | $2,828.53 | Candidate Baseline |

*Selection Criterion:* Winning model automatically selected based on highest test $R^2$ score and lowest test RMSE.

---

## 📁 Project Structure

```
Insurance-Cost-AI/
├── app.py                            # Application entry point with Streamlit navigation
├── requirements.txt                  # Python dependencies
├── README.md                         # Comprehensive project documentation
├── .gitignore                        # Git exclusion rules
│
├── data/
│   ├── raw/
│   │   └── insurance_dataset.csv     # Raw 6,000-sample synthetic dataset
│   └── processed/
│       └── insurance_processed.csv   # Feature-engineered processed dataset
│
├── models/
│   ├── best_model.pkl                # Serialized winning production model
│   ├── preprocessing_pipeline.pkl   # Fitted Scikit-Learn ColumnTransformer
│   ├── all_models.pkl               # Serialized candidate models
│   └── model_evaluation_report.json # Comprehensive evaluation metrics & diagnostics
│
├── src/
│   ├── data_processing/
│   │   ├── generate_dataset.py       # Actuarial synthetic data generation script
│   │   └── preprocess.py             # Feature engineering & preprocessing pipeline
│   │
│   ├── models/
│   │   ├── evaluate_models.py        # Actuarial metric calculation (MAE, RMSE, R²)
│   │   └── train_models.py           # Multi-model training, benchmarking & selection
│   │
│   ├── prediction/
│   │   └── prediction_helper.py      # Real-time inference & explanation engine
│   │
│   └── analytics/
│       └── analytics_helper.py       # Plotly chart generators & KPI calculators
│
├── pages/
│   ├── dashboard.py                  # Executive Dashboard with KPI cards & filters
│   ├── predictor.py                  # Insurance Cost Predictor & Insights
│   ├── risk_profile.py               # Customer Risk Profile & Gauge Visualizer
│   ├── analytics.py                  # Cost & Demographic Analytics
│   ├── model_performance.py          # Model Benchmarks, Residuals & Explainability
│   ├── scenario_analysis.py          # Multi-Cohort & Real-Time What-If Simulator
│   └── data_explorer.py              # Data Audit, Statistics & Custom Visualizer
│
└── assets/
    └── style.css                     # Custom styling, Inter typography & glassmorphism
```

---

## 💻 Installation & Setup (Windows)

Open PowerShell or Command Prompt:

### 1. Clone or Navigate to the Project Folder
```powershell
cd Healthcare_Insurance/Insurance-Cost-AI
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## ⚙️ Model Training Instructions

To execute the data generation, feature engineering, and model training pipeline from scratch:

```powershell
python src/models/train_models.py
```

This script will:
1. Generate the 6,000-record synthetic insurance dataset in `data/raw/`.
2. Apply feature engineering and save the processed dataset to `data/processed/`.
3. Train Linear Regression, Random Forest, and Gradient Boosting models.
4. Evaluate test metrics, pick the winning model, and save artifacts in `models/`.

---

## 🖥️ Running the Application

Launch the Streamlit web application:

```powershell
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## ⚖️ Risk Logic & Non-Causal Disclaimers

### Explanation Wording Standard
In accordance with ethical AI guidelines, feature influences are strictly designated as:
> *"associated with the model prediction"*

No claims of direct causality or absolute clinical etiology are made.

### Underwriting Advisory Notice
All risk indices, scores, and dollar projections are generated via predictive regression models for estimation, comparative benchmarking, and educational demonstration. They do **not** constitute formal underwriting decisions, binding policy quotes, or clinical medical evaluations.

---

## 🔮 Limitations & Future Enhancements

### Current Limitations
- Operates on a synthetic cohort generated to simulate realistic actuarial relationships.
- Does not incorporate longitudinal electronic health record (EHR) time-series data or detailed prescription drug histories.

### Future Enhancements
- Integration of geospatial zip-code level socio-economic health indices.
- Support for family plan joint-underwriting risk pooling simulations.
- Automated API endpoints via FastAPI for microservice integration.
