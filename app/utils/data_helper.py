import os
import pandas as pd
import numpy as np
import streamlit as st

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "EMI_dataset.csv")

@st.cache_data(show_spinner="Loading EMI Financial Dataset...")
def load_raw_data(sample_n: int = None) -> pd.DataFrame:
    """Loads the official EMI dataset from the data folder."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    if sample_n and sample_n < len(df):
        return df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans data without leakage for analysis."""
    data = df.copy()
    
    # Clean age
    data['age'] = data['age'].astype(str).str.replace(r'(\.\d+)\.\d+$', r'\1', regex=True)
    data['age'] = pd.to_numeric(data['age'], errors='coerce')
    data['age'] = data['age'].fillna(35).clip(18, 75)
    
    # Standardize gender
    gender_map = {
        'Male': 'Male', 'MALE': 'Male', 'M': 'Male', 'male': 'Male',
        'Female': 'Female', 'FEMALE': 'Female', 'F': 'Female', 'female': 'Female'
    }
    data['gender'] = data['gender'].map(gender_map).fillna('Male')
    
    # Clean numeric fields
    data['monthly_salary'] = data['monthly_salary'].astype(str).str.replace(r'(\.\d+)\.\d+$', r'\1', regex=True)
    data['monthly_salary'] = pd.to_numeric(data['monthly_salary'], errors='coerce').fillna(50000).clip(lower=5000)
    
    data['bank_balance'] = data['bank_balance'].astype(str).str.replace(r'(\.\d+)\.\d+$', r'\1', regex=True)
    data['bank_balance'] = pd.to_numeric(data['bank_balance'], errors='coerce').fillna(100000).clip(lower=0)
    
    data['monthly_rent'] = pd.to_numeric(data['monthly_rent'], errors='coerce').fillna(0).clip(lower=0)
    data['credit_score'] = pd.to_numeric(data['credit_score'], errors='coerce').fillna(700).clip(300, 900)
    data['emergency_fund'] = pd.to_numeric(data['emergency_fund'], errors='coerce').fillna(90000).clip(lower=0)
    
    # Categoricals
    data['education'] = data['education'].fillna('Graduate')
    data['marital_status'] = data['marital_status'].fillna('Married')
    data['employment_type'] = data['employment_type'].fillna('Private')
    data['company_type'] = data['company_type'].fillna('Mid-size')
    data['house_type'] = data['house_type'].fillna('Rented')
    data['existing_loans'] = data['existing_loans'].fillna('No')
    data['emi_scenario'] = data['emi_scenario'].fillna('Personal Loan EMI')
    
    return data

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes standard financial health indicators."""
    data = df.copy()
    data['total_monthly_expenses'] = (
        data['school_fees'] + data['college_fees'] + 
        data['travel_expenses'] + data['groceries_utilities'] + 
        data['other_monthly_expenses'] + data['monthly_rent']
    )
    data['total_financial_obligations'] = data['total_monthly_expenses'] + data['current_emi_amount']
    data['disposable_income'] = data['monthly_salary'] - data['total_financial_obligations']
    data['expense_to_income_ratio'] = (data['total_monthly_expenses'] / (data['monthly_salary'] + 1.0)).clip(0, 5)
    data['debt_to_income_ratio'] = (data['current_emi_amount'] / (data['monthly_salary'] + 1.0)).clip(0, 3)
    data['savings_ratio'] = (data['bank_balance'] / (data['monthly_salary'] + 1.0)).clip(0, 20)
    data['emergency_fund_ratio'] = (data['emergency_fund'] / (data['total_monthly_expenses'] + 1.0)).clip(0, 30)
    
    norm_cibil = (data['credit_score'] - 300) / 600.0
    norm_exp = np.clip(data['years_of_employment'] / 15.0, 0, 1.0)
    norm_emergency = np.clip(data['emergency_fund'] / (data['total_monthly_expenses'] * 6 + 1.0), 0, 1.0)
    data['financial_stability_score'] = ((norm_cibil * 0.45 + norm_exp * 0.25 + norm_emergency * 0.30) * 100).round(2)
    return data
