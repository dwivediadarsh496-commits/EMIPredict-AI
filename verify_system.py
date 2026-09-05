import os
import sys
import pandas as pd

# Add app directory to sys.path
sys.path.insert(0, r"d:\EMIPredict AI")

print("--- 1. Testing Model Loading & Inference ---")
from app.utils.model_helper import load_models, predict_applicant
class_pipe, reg_pipe, meta = load_models()
print("Models loaded successfully!")
print(f"Classification classes: {meta.get('classification', {}).get('classes')}")
print(f"Regression best model: {meta.get('regression', {}).get('best_model')}")

test_profile = {
    'age': 29.0,
    'gender': 'Female',
    'marital_status': 'Single',
    'education': 'Graduate',
    'monthly_salary': 55000.0,
    'employment_type': 'Private',
    'years_of_employment': 4.5,
    'company_type': 'MNC',
    'house_type': 'Rented',
    'monthly_rent': 10000.0,
    'family_size': 2,
    'dependents': 0,
    'school_fees': 0.0,
    'college_fees': 0.0,
    'travel_expenses': 4000.0,
    'groceries_utilities': 10000.0,
    'other_monthly_expenses': 5000.0,
    'existing_loans': 'No',
    'current_emi_amount': 0.0,
    'credit_score': 745.0,
    'bank_balance': 90000.0,
    'emergency_fund': 80000.0,
    'emi_scenario': 'E-commerce Shopping EMI',
    'requested_amount': 45000.0,
    'requested_tenure': 12
}

res = predict_applicant(test_profile)
print("Inference Result:", res)
assert "eligibility" in res
assert "max_safe_emi" in res

print("\n--- 2. Testing SQLite Database CRUD ---")
from app.utils.db_helper import create_applicant, get_all_applicants, update_applicant, delete_applicant

df_apps = get_all_applicants()
initial_count = len(df_apps)
print(f"Initial DB record count: {initial_count}")

# Create
new_id = create_applicant({
    "applicant_name": "Test User",
    "monthly_salary": 60000.0,
    "credit_score": 750.0,
    "emi_scenario": "Personal Loan EMI"
})
print(f"Created test applicant ID: {new_id}")

# Read
df_apps_after = get_all_applicants()
assert len(df_apps_after) == initial_count + 1

# Update
update_success = update_applicant(new_id, {"status": "Approved"})
assert update_success

# Delete
delete_success = delete_applicant(new_id)
assert delete_success
print(f"Deleted test applicant ID: {new_id}")
print(f"Final DB record count: {len(get_all_applicants())}")

print("\n--- 3. Testing Data Helper ---")
from app.utils.data_helper import load_raw_data
df_sample = load_raw_data(sample_n=1000)
print(f"Sample data loaded successfully with shape: {df_sample.shape}")

print("\n>>> ALL SYSTEM MODULES & TESTS PASSED WITH 100% SUCCESS! <<<")
