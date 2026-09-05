import os
import sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.components.ui import apply_fintech_style, render_header, render_metric_card, render_disclaimer
from app.utils.db_helper import get_all_applicants, create_applicant, update_applicant, delete_applicant, get_applicant_by_id

st.set_page_config(page_title="Data Management - EMIPredict AI", page_icon="🗄️", layout="wide")
apply_fintech_style()

render_header(
    title="Financial Data Management (CRUD)",
    subtitle="Create, read, update, and delete applicant records stored in local SQLite database.",
    badge="Database Operations"
)

tab_view, tab_create, tab_edit, tab_delete = st.tabs([
    "📋 View Records (READ)",
    "➕ New Applicant (CREATE)",
    "✏️ Edit Record (UPDATE)",
    "🗑️ Remove Record (DELETE)"
])

# 1. READ
with tab_view:
    st.subheader("Applicant Applications Database")
    df_apps = get_all_applicants()
    if not df_apps.empty:
        st.dataframe(df_apps, use_container_width=True, hide_index=True)
        st.caption(f"Total stored records: {len(df_apps)}")
    else:
        st.info("No records currently exist in the database.")

# 2. CREATE
with tab_create:
    st.subheader("Add New Applicant Record")
    with st.form("create_applicant_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Applicant Name", value="Vikas Mehta")
            age = st.number_input("Age", 18, 75, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital = st.selectbox("Marital Status", ["Married", "Single"])
            salary = st.number_input("Monthly Salary (₹)", 10000.0, 500000.0, 55000.0, 5000.0)
        with col2:
            credit = st.number_input("Credit Score (CIBIL)", 300.0, 900.0, 740.0, 10.0)
            scenario = st.selectbox("EMI Scenario", ["Personal Loan EMI", "Vehicle EMI", "Education EMI", "Home Appliances EMI", "E-commerce Shopping EMI"])
            req_amount = st.number_input("Requested Loan (₹)", 10000.0, 2000000.0, 150000.0, 10000.0)
            tenure = st.number_input("Tenure (Months)", 3, 84, 24, 3)
            status = st.selectbox("Underwriting Status", ["Under Review", "Approved", "Rejected"])
            
        submitted = st.form_submit_button("➕ Create Applicant Record", type="primary")
        if submitted:
            new_data = {
                "applicant_name": name,
                "age": age,
                "gender": gender,
                "marital_status": marital,
                "monthly_salary": salary,
                "credit_score": credit,
                "emi_scenario": scenario,
                "requested_amount": req_amount,
                "requested_tenure": tenure,
                "predicted_eligibility": "Pending Assessment",
                "predicted_max_emi": round(salary * 0.25, 2),
                "status": status
            }
            new_id = create_applicant(new_data)
            st.success(f"Applicant '{name}' successfully created with Record ID: #{new_id}!")
            st.rerun()

# 3. UPDATE
with tab_edit:
    st.subheader("Update Existing Record")
    df_apps = get_all_applicants()
    if not df_apps.empty:
        record_ids = df_apps["id"].tolist()
        selected_id = st.selectbox("Select Record ID to Edit:", options=record_ids)
        
        current_record = get_applicant_by_id(selected_id)
        if current_record:
            with st.form("edit_applicant_form"):
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    e_name = st.text_input("Name", value=current_record["applicant_name"])
                    e_salary = st.number_input("Monthly Salary (₹)", value=float(current_record["monthly_salary"]))
                    e_credit = st.number_input("Credit Score", value=float(current_record["credit_score"]))
                with col_e2:
                    e_amount = st.number_input("Requested Amount (₹)", value=float(current_record["requested_amount"]))
                    e_status = st.selectbox("Status", ["Approved", "Under Review", "Rejected"], 
                                            index=["Approved", "Under Review", "Rejected"].index(current_record.get("status", "Under Review")))
                    e_eligibility = st.selectbox("Eligibility Classification", ["Eligible", "High_Risk", "Not_Eligible", "Pending Assessment"],
                                                 index=["Eligible", "High_Risk", "Not_Eligible", "Pending Assessment"].index(current_record.get("predicted_eligibility", "Pending Assessment")))
                
                update_btn = st.form_submit_button("💾 Save Changes", type="primary")
                if update_btn:
                    updates = {
                        "applicant_name": e_name,
                        "monthly_salary": e_salary,
                        "credit_score": e_credit,
                        "requested_amount": e_amount,
                        "status": e_status,
                        "predicted_eligibility": e_eligibility
                    }
                    update_applicant(selected_id, updates)
                    st.success(f"Record #{selected_id} updated successfully!")
                    st.rerun()
    else:
        st.info("No records available to edit.")

# 4. DELETE
with tab_delete:
    st.subheader("Delete Applicant Record")
    df_apps = get_all_applicants()
    if not df_apps.empty:
        del_id = st.selectbox("Select Record ID to Delete:", options=df_apps["id"].tolist())
        del_applicant = get_applicant_by_id(del_id)
        
        if del_applicant:
            st.warning(f"Are you sure you want to permanently delete record #{del_id} for **{del_applicant['applicant_name']}**?")
            if st.button("🗑️ Confirm Delete Record", type="primary"):
                delete_applicant(del_id)
                st.success(f"Record #{del_id} permanently deleted.")
                st.rerun()
    else:
        st.info("No records available to delete.")

render_disclaimer()
