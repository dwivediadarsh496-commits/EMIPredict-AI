import os
import sqlite3
import pandas as pd
from datetime import datetime

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "financial_records.db")

def get_connection():
    """Returns a SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite applicant records table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS applicants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        applicant_name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        marital_status TEXT,
        monthly_salary REAL,
        credit_score REAL,
        emi_scenario TEXT,
        requested_amount REAL,
        requested_tenure INTEGER,
        predicted_eligibility TEXT,
        predicted_max_emi REAL,
        status TEXT DEFAULT 'Under Review',
        created_at TEXT
    )
    """)
    conn.commit()
    
    # Pre-populate sample records if empty
    cursor.execute("SELECT COUNT(*) FROM applicants")
    count = cursor.fetchone()[0]
    if count == 0:
        sample_records = [
            ("Aarav Sharma", 34, "Male", "Married", 75000.0, 780.0, "Personal Loan EMI", 300000.0, 24, "Eligible", 18500.0, "Approved", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("Priya Nair", 28, "Female", "Single", 45000.0, 690.0, "E-commerce Shopping EMI", 40000.0, 12, "Eligible", 8200.0, "Approved", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("Rohit Verma", 42, "Male", "Married", 38000.0, 560.0, "Vehicle EMI", 450000.0, 36, "Not_Eligible", 2400.0, "Rejected", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("Ananya Patel", 26, "Female", "Single", 60000.0, 640.0, "Home Appliances EMI", 85000.0, 18, "High_Risk", 7100.0, "Under Review", datetime.now().strftime("%Y-%m-%d %H:%M"))
        ]
        cursor.executemany("""
        INSERT INTO applicants (applicant_name, age, gender, marital_status, monthly_salary, credit_score, emi_scenario, requested_amount, requested_tenure, predicted_eligibility, predicted_max_emi, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_records)
        conn.commit()
        
    conn.close()

def create_applicant(data: dict) -> int:
    """Creates a new financial applicant record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO applicants (applicant_name, age, gender, marital_status, monthly_salary, credit_score, emi_scenario, requested_amount, requested_tenure, predicted_eligibility, predicted_max_emi, status, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("applicant_name", "Anonymous Applicant"),
        data.get("age", 30),
        data.get("gender", "Male"),
        data.get("marital_status", "Married"),
        data.get("monthly_salary", 50000.0),
        data.get("credit_score", 700.0),
        data.get("emi_scenario", "Personal Loan EMI"),
        data.get("requested_amount", 100000.0),
        data.get("requested_tenure", 24),
        data.get("predicted_eligibility", "Under Review"),
        data.get("predicted_max_emi", 0.0),
        data.get("status", "Under Review"),
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id

def get_all_applicants() -> pd.DataFrame:
    """Fetches all applicant records as a pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM applicants ORDER BY id DESC", conn)
    conn.close()
    return df

def get_applicant_by_id(record_id: int) -> dict:
    """Fetches a single applicant record by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM applicants WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def update_applicant(record_id: int, updates: dict) -> bool:
    """Updates an applicant record."""
    conn = get_connection()
    cursor = conn.cursor()
    fields = []
    values = []
    for k, v in updates.items():
        fields.append(f"{k} = ?")
        values.append(v)
    values.append(record_id)
    
    query = f"UPDATE applicants SET {', '.join(fields)} WHERE id = ?"
    cursor.execute(query, values)
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success

def delete_applicant(record_id: int) -> bool:
    """Deletes an applicant record by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM applicants WHERE id = ?", (record_id,))
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success

# Initialize table on import
init_db()
