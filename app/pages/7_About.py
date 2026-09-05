import os
import sys
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.components.ui import apply_fintech_style, render_header, render_disclaimer

st.set_page_config(page_title="About Project - EMIPredict AI", page_icon="ℹ️", layout="wide")
apply_fintech_style()

render_header(
    title="About EMIPredict AI",
    subtitle="Intelligent Financial Risk Assessment & Sustainable EMI Underwriting Platform",
    badge="Project Documentation"
)

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    ### 🎯 Project Overview & Objective
    **EMIPredict AI** is an industrial-grade Machine Learning decision-support platform designed to solve retail credit risk underwriting and sustainable monthly installment (EMI) sizing. Built as an end-to-end B.Tech / internship capstone project, it models real-world retail banking applications across 5 primary loan categories:
    1. **E-commerce Shopping EMI**
    2. **Home Appliances EMI**
    3. **Vehicle Financing EMI**
    4. **Unsecured Personal Loan EMI**
    5. **Higher Education EMI**

    ---

    ### 🏗️ Architectural Pipeline
    ```
    Raw Financial Records (404,800 Rows)
                    │
                    ▼
          Data Quality Assessment
        (Formatting & Type Cleansing)
                    │
                    ▼
           Feature Engineering
    (DTI, Cash Flow, Stability Index)
                    │
           ┌────────┴────────┐
           ▼                 ▼
    Classification      Regression
       Engine             Engine
     (Eligibility)      (Max Safe EMI)
           │                 │
           └────────┬────────┘
                    ▼
             MLflow Registry
           & Saved Pipelines
                    │
                    ▼
         Streamlit FinTech App
     (Predictions, Analytics, CRUD)
    ```

    ---

    ### 🤖 Machine Learning Methodology
    - **Preprocessing**: `ColumnTransformer` with `StandardScaler` for numeric columns and `OneHotEncoder(handle_unknown='ignore')` for categorical columns.
    - **Classification Target**: `emi_eligibility` (`Eligible`, `High_Risk`, `Not_Eligible`).
      - Champion Model: **XGBoost Classifier** (95.91% Accuracy, 0.9491 F1-score, 0.9965 ROC-AUC).
    - **Regression Target**: `max_monthly_emi` (Continuous monthly safe EMI capacity in INR).
      - Champion Model: **XGBoost Regressor** (RMSE ₹716.18, R² 0.9916, MAE ₹219.45).
    - **MLflow Tracking**: Complete hyperparameter and evaluation metrics logged to SQLite tracking database.
    """)

with col2:
    st.markdown("""
    ### 💻 Technology Stack
    - **Core Runtime**: Python 3.10
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn
    - **Machine Learning**: Scikit-Learn, XGBoost
    - **Model Tracking**: MLflow (SQLite backend)
    - **Web Framework**: Streamlit (Multi-page app)
    - **Persistence**: Joblib & SQLite3

    ---

    ### 🏢 Business Use Cases
    - **Digital NBFCs**: Sub-second pre-qualification during checkout without hard credit pulls.
    - **Retail Banks**: Automated credit risk scoring and portfolio health monitoring.
    - **Loan Officers**: Transparent decision-support breakdown detailing disposable income and expense burdens.

    ---

    ### ☁️ Cloud Deployment Readiness
    - Configured for Streamlit Cloud deployment with relative paths.
    - Zero local-only hardcoded file dependencies.
    - Isolated SQLite application database.
    """)

render_disclaimer()
