import os
import sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.components.ui import apply_fintech_style, render_header, render_metric_card, render_disclaimer
from app.utils.model_helper import predict_applicant
from app.utils.db_helper import create_applicant

st.set_page_config(page_title="EMI Prediction - EMIPredict AI", page_icon="⚡", layout="wide")
apply_fintech_style()

render_header(
    title="Real-Time EMI Prediction",
    subtitle="Evaluate applicant eligibility and safe monthly EMI threshold using production ML models.",
    badge="Inference Engine"
)

with st.form("emi_prediction_form"):
    st.subheader("1. Applicant Demographics & Employment")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        applicant_name = st.text_input("Applicant Full Name", value="Rahul Deshmukh")
        age = st.number_input("Age", min_value=18, max_value=75, value=32, step=1)
    with col2:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        marital_status = st.selectbox("Marital Status", options=["Married", "Single"])
    with col3:
        education = st.selectbox("Education Level", options=["Graduate", "Post Graduate", "Professional", "High School"])
        employment_type = st.selectbox("Employment Type", options=["Private", "Government", "Self-employed"])
    with col4:
        years_of_employment = st.number_input("Years of Employment", min_value=0.0, max_value=45.0, value=6.0, step=0.5)
        company_type = st.selectbox("Company Type", options=["Large Indian", "MNC", "Mid-size", "Startup", "Small"])

    st.subheader("2. Income, Living & Family Commitments")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        monthly_salary = st.number_input("Monthly Salary (₹)", min_value=5000.0, max_value=1000000.0, value=65000.0, step=2000.0)
        house_type = st.selectbox("House Type", options=["Rented", "Own", "Family"])
    with col6:
        monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0.0, max_value=200000.0, value=12000.0, step=1000.0)
        family_size = st.number_input("Family Size", min_value=1, max_value=15, value=3, step=1)
    with col7:
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1, step=1)
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", min_value=0.0, max_value=150000.0, value=12000.0, step=1000.0)
    with col8:
        travel_expenses = st.number_input("Travel Expenses (₹)", min_value=0.0, max_value=100000.0, value=5000.0, step=500.0)
        other_monthly_expenses = st.number_input("Other Expenses (₹)", min_value=0.0, max_value=100000.0, value=6000.0, step=500.0)

    st.subheader("3. Education Fees, Credit History & Current Loans")
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        school_fees = st.number_input("School Fees (₹)", min_value=0.0, max_value=100000.0, value=4000.0, step=500.0)
        college_fees = st.number_input("College Fees (₹)", min_value=0.0, max_value=150000.0, value=0.0, step=1000.0)
    with col10:
        existing_loans = st.selectbox("Existing Loans?", options=["No", "Yes"])
        current_emi_amount = st.number_input("Current Ongoing EMI (₹)", min_value=0.0, max_value=200000.0, value=0.0, step=1000.0)
    with col11:
        credit_score = st.number_input("Credit Score (CIBIL 300-900)", min_value=300.0, max_value=900.0, value=760.0, step=10.0)
        bank_balance = st.number_input("Bank Balance (₹)", min_value=0.0, max_value=5000000.0, value=140000.0, step=10000.0)
    with col12:
        emergency_fund = st.number_input("Emergency Reserve Fund (₹)", min_value=0.0, max_value=5000000.0, value=120000.0, step=10000.0)
        emi_scenario = st.selectbox("Loan / EMI Scenario", options=[
            "Personal Loan EMI", "E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Education EMI"
        ])

    st.subheader("4. Requested Financing Details")
    col13, col14, col15 = st.columns(3)
    with col13:
        requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=5000.0, max_value=5000000.0, value=250000.0, step=10000.0)
    with col14:
        requested_tenure = st.number_input("Requested Tenure (Months)", min_value=3, max_value=120, value=24, step=3)
    with col15:
        st.markdown("<br>", unsafe_allow_html=True)
        approx_req_emi = requested_amount / max(requested_tenure, 1)
        st.info(f"Estimated Monthly Payment: **₹{approx_req_emi:,.2f}**")

    submit_btn = st.form_submit_button("⚡ Run Full AI Risk & EMI Assessment", use_container_width=True, type="primary")

if submit_btn:
    profile = {
        'age': float(age),
        'gender': gender,
        'marital_status': marital_status,
        'education': education,
        'monthly_salary': float(monthly_salary),
        'employment_type': employment_type,
        'years_of_employment': float(years_of_employment),
        'company_type': company_type,
        'house_type': house_type,
        'monthly_rent': float(monthly_rent),
        'family_size': int(family_size),
        'dependents': int(dependents),
        'school_fees': float(school_fees),
        'college_fees': float(college_fees),
        'travel_expenses': float(travel_expenses),
        'groceries_utilities': float(groceries_utilities),
        'other_monthly_expenses': float(other_monthly_expenses),
        'existing_loans': existing_loans,
        'current_emi_amount': float(current_emi_amount),
        'credit_score': float(credit_score),
        'bank_balance': float(bank_balance),
        'emergency_fund': float(emergency_fund),
        'emi_scenario': emi_scenario,
        'requested_amount': float(requested_amount),
        'requested_tenure': int(requested_tenure)
    }
    
    with st.spinner("Analyzing applicant financial profile through ML pipelines..."):
        res = predict_applicant(profile)
        
    st.markdown("### 🎯 AI Underwriting Assessment Results")
    
    # Result cards
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        status = res["eligibility"]
        badge_class = "badge-eligible" if status == "Eligible" else ("badge-highrisk" if status == "High_Risk" else "badge-noteligible")
        st.markdown(f"""
        <div class="fintech-card" style="border-left: 4px solid {'#10b981' if status == 'Eligible' else ('#f59e0b' if status == 'High_Risk' else '#ef4444')};">
            <div class="fintech-card-label">Predicted EMI Eligibility</div>
            <div class="fintech-card-value"><span class="{badge_class}">{status}</span></div>
            <div class="fintech-card-sub">Model Confidence: <strong>{res['confidence']}%</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        safe_emi = res["max_safe_emi"]
        st.markdown(f"""
        <div class="fintech-card" style="border-left: 4px solid #3b82f6;">
            <div class="fintech-card-label">Maximum Safe Monthly EMI</div>
            <div class="fintech-card-value" style="color: #60a5fa;">₹{safe_emi:,.2f}</div>
            <div class="fintech-card-sub">Recommended Safe Ceiling</div>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col3:
        affordability = res["affordability"]
        aff_color = "#10b981" if affordability == "Affordable" else "#ef4444"
        st.markdown(f"""
        <div class="fintech-card" style="border-left: 4px solid {aff_color};">
            <div class="fintech-card-label">Requested Loan Affordability</div>
            <div class="fintech-card-value" style="color: {aff_color};">{affordability}</div>
            <div class="fintech-card-sub">Req: ₹{res['requested_monthly_emi']:,.0f} vs Safe: ₹{safe_emi:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Diagnostic Metrics Row
    st.markdown("#### 🔍 Financial Health Breakdown")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Disposable Monthly Income", f"₹{res['disposable_income']:,.2f}")
    m2.metric("Total Living Obligations", f"₹{res['total_expenses']:,.2f}")
    m3.metric("Debt-to-Income (DTI)", f"{res['debt_to_income_ratio']:.1f}%")
    m4.metric("Financial Stability Score", f"{res['financial_stability_score']}/100")
    
    # Save to Database option
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("💾 Save Record to Applicant Database", type="secondary"):
        new_record = {
            "applicant_name": applicant_name,
            "age": int(age),
            "gender": gender,
            "marital_status": marital_status,
            "monthly_salary": float(monthly_salary),
            "credit_score": float(credit_score),
            "emi_scenario": emi_scenario,
            "requested_amount": float(requested_amount),
            "requested_tenure": int(requested_tenure),
            "predicted_eligibility": res["eligibility"],
            "predicted_max_emi": res["max_safe_emi"],
            "status": "Approved" if res["eligibility"] == "Eligible" else ("Under Review" if res["eligibility"] == "High_Risk" else "Rejected")
        }
        rec_id = create_applicant(new_record)
        st.success(f"Applicant record saved to database with Record ID: #{rec_id}!")

render_disclaimer()
