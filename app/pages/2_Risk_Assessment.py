import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.components.ui import apply_fintech_style, render_header, render_metric_card, render_disclaimer
from app.utils.model_helper import predict_applicant

st.set_page_config(page_title="Financial Risk Assessment - EMIPredict AI", page_icon="🛡️", layout="wide")
apply_fintech_style()

render_header(
    title="Financial Risk Assessment",
    subtitle="In-depth stress testing, debt burden diagnostics, and risk tier categorization.",
    badge="Risk Analytics"
)

st.markdown("""
Evaluate the applicant's risk tolerance under varying economic stress scenarios and inspect key ratios including Debt-to-Income (DTI), living obligations, and liquid reserve adequacy.
""")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Applicant Profile Setup")
    salary = st.slider("Monthly Gross Salary (₹)", 15000, 300000, 75000, 5000)
    current_emi = st.slider("Current Ongoing EMI Obligations (₹)", 0, 100000, 12000, 2000)
    living_expenses = st.slider("Living & Household Expenses (₹)", 10000, 150000, 35000, 2500)
    cibil = st.slider("Credit Score (CIBIL)", 300, 900, 720, 10)
    emergency_fund = st.slider("Liquid Emergency Reserves (₹)", 0, 500000, 100000, 10000)
    
    st.markdown("#### Financing Request")
    loan_amount = st.number_input("Requested Loan Principal (₹)", min_value=10000, max_value=2000000, value=300000, step=25000)
    tenure_months = st.slider("Loan Tenure (Months)", 6, 84, 24, 6)
    scenario = st.selectbox("Scenario", ["Personal Loan EMI", "Vehicle EMI", "Education EMI", "Home Appliances EMI", "E-commerce Shopping EMI"])

with col_right:
    st.subheader("AI Multi-Dimensional Risk Diagnostic")
    
    profile = {
        'age': 35.0,
        'gender': 'Male',
        'marital_status': 'Married',
        'education': 'Graduate',
        'monthly_salary': float(salary),
        'employment_type': 'Private',
        'years_of_employment': 7.0,
        'company_type': 'Mid-size',
        'house_type': 'Rented',
        'monthly_rent': float(living_expenses * 0.35),
        'family_size': 3,
        'dependents': 1,
        'school_fees': float(living_expenses * 0.15),
        'college_fees': 0.0,
        'travel_expenses': float(living_expenses * 0.15),
        'groceries_utilities': float(living_expenses * 0.25),
        'other_monthly_expenses': float(living_expenses * 0.10),
        'existing_loans': 'Yes' if current_emi > 0 else 'No',
        'current_emi_amount': float(current_emi),
        'credit_score': float(cibil),
        'bank_balance': float(emergency_fund * 1.2),
        'emergency_fund': float(emergency_fund),
        'emi_scenario': scenario,
        'requested_amount': float(loan_amount),
        'requested_tenure': int(tenure_months)
    }
    
    res = predict_applicant(profile)
    
    # Key Indicator Gauges
    g1, g2, g3 = st.columns(3)
    
    with g1:
        dti = (current_emi / max(salary, 1)) * 100
        dti_color = "#10b981" if dti <= 35 else ("#f59e0b" if dti <= 50 else "#ef4444")
        st.markdown(f"""
        <div class="fintech-card" style="border-left: 4px solid {dti_color};">
            <div class="fintech-card-label">Debt-to-Income (DTI)</div>
            <div class="fintech-card-value" style="color: {dti_color};">{dti:.1f}%</div>
            <div class="fintech-card-sub">Industry Cap: 40.0%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with g2:
        stability = res["financial_stability_score"]
        stab_color = "#10b981" if stability >= 70 else ("#f59e0b" if stability >= 50 else "#ef4444")
        st.markdown(f"""
        <div class="fintech-card" style="border-left: 4px solid {stab_color};">
            <div class="fintech-card-label">Stability Index</div>
            <div class="fintech-card-value" style="color: {stab_color};">{stability}/100</div>
            <div class="fintech-card-sub">Reserve & Credit Health</div>
        </div>
        """, unsafe_allow_html=True)
        
    with g3:
        reserve_months = emergency_fund / max(living_expenses + current_emi, 1)
        res_color = "#10b981" if reserve_months >= 3 else ("#f59e0b" if reserve_months >= 1 else "#ef4444")
        st.markdown(f"""
        <div class="fintech-card" style="border-left: 4px solid {res_color};">
            <div class="fintech-card-label">Emergency Coverage</div>
            <div class="fintech-card-value" style="color: {res_color};">{reserve_months:.1f} mo</div>
            <div class="fintech-card-sub">Benchmark: >= 3 Months</div>
        </div>
        """, unsafe_allow_html=True)

    # Cashflow Breakdown Bar
    st.markdown("#### 💵 Monthly Income Allocation Breakdown")
    obligations = current_emi + living_expenses
    disposable = salary - obligations
    
    df_cashflow = pd.DataFrame({
        "Category": ["Living Expenses", "Ongoing EMI Debt", "Remaining Disposable"],
        "Amount (₹)": [living_expenses, current_emi, max(disposable, 0)]
    })
    
    st.bar_chart(df_cashflow.set_index("Category"), color="#3b82f6")
    
    # Recommendation Summary Box
    st.markdown("#### 📋 AI Underwriting Assessment Summary")
    if res["eligibility"] == "Eligible":
        st.success(f"""
        ✅ **Low Risk / Approved Profile**: Applicant qualifies for lending with high confidence ({res['confidence']}%).  
        - Recommended Max Monthly EMI: **₹{res['max_safe_emi']:,.2f}**  
        - Requested Monthly Installment: **₹{loan_amount/tenure_months:,.2f}**  
        - Available Discretionary Headroom: **₹{max(disposable, 0):,.2f} / month**
        """)
    elif res["eligibility"] == "High_Risk":
        st.warning(f"""
        ⚠️ **Elevated Financial Risk**: Applicant exhibits debt strain or narrow reserve margins.  
        - Maximum Sustainable EMI: **₹{res['max_safe_emi']:,.2f}**  
        - **Mitigation Strategy**: Recommend extending loan tenure to {min(tenure_months + 12, 60)} months or providing guarantor collateral.
        """)
    else:
        st.error(f"""
        ❌ **High Probability of Default / Non-Eligible**: Current living expenses and commitments exceed sustainable underwriting parameters.  
        - Estimated Safe Ceiling: **₹{res['max_safe_emi']:,.2f}**  
        - **Underwriting Note**: High debt-to-income or low disposable cash flow prohibits new credit origination.
        """)

render_disclaimer()
