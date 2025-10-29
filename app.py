import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Load Model and Scaler
# ----------------------------
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ----------------------------
# Database Setup
# ----------------------------
conn = sqlite3.connect("diabetes_records.db", check_same_thread=False)
c = conn.cursor()

# Create table with username if it doesn’t exist
# Create table with all required columns (if not exists)
c.execute("""
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    Pregnancies INTEGER,
    Glucose REAL,
    BloodPressure REAL,
    SkinThickness REAL,
    Insulin REAL,
    BMI REAL,
    DiabetesPedigreeFunction REAL,
    Age INTEGER,
    Result TEXT
)
""")
conn.commit()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Smart Diabetes Predictor", layout="centered")
st.title("🩺 Smart Diabetes Predictor")

menu = ["🏠 Predict Diabetes", "📋 View Records", "🗑️ Delete Record"]
choice = st.sidebar.selectbox("Navigation", menu)

# ----------------------------
# Normal Medical Ranges
# ----------------------------
normal_ranges = {
    "Pregnancies": "0 - 10",
    "Glucose": "70 - 140 mg/dL",
    "BloodPressure": "80 - 120 mmHg",
    "SkinThickness": "10 - 50 mm",
    "Insulin": "15 - 276 µU/mL",
    "BMI": "18.5 - 24.9",
    "DiabetesPedigreeFunction": "0.0 - 1.0",
    "Age": "20 - 80 years"
}

# ----------------------------
# 🏠 Predict Diabetes Section
# ----------------------------
if choice == "🏠 Predict Diabetes":
    st.header("Enter Patient Details")

    username = st.text_input("👤 Username")
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        Glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 100)
        BloodPressure = st.number_input("Blood Pressure (mmHg)", 0, 200, 80)
        SkinThickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)

    with col2:
        Insulin = st.number_input("Insulin Level (µU/mL)", 0, 900, 85)
        BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        Age = st.number_input("Age", 1, 120, 30)

    if st.button("🔍 Predict"):
        if username.strip() == "":
            st.warning("Please enter a username.")
        else:
            input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                    Insulin, BMI, DiabetesPedigreeFunction, Age]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            result_label = "🩸 Diabetes Detected" if prediction == 1 else "✅ No Diabetes Detected"
            st.subheader(f"Result: {result_label}")

            # Show normal medical ranges
            st.markdown("### 📊 Normal Medical Ranges")
            for key, value in normal_ranges.items():
                st.write(f"**{key}:** {value}")

            # Save record
            c.execute("""
            INSERT INTO records (username, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                 DiabetesPedigreeFunction, Age, Result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (username, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                  DiabetesPedigreeFunction, Age, result_label))
            conn.commit()
            st.success("Record saved successfully!")

# ----------------------------
# 📋 View Records Section
# ----------------------------
elif choice == "📋 View Records":
    st.header("Patient Records (Oldest First)")
    df = pd.read_sql_query("SELECT * FROM records ORDER BY id ASC", conn)
    st.dataframe(df)

# ----------------------------
# 🗑️ Delete Record Section
# ----------------------------
elif choice == "🗑️ Delete Record":
    st.header("Delete Record")

    df = pd.read_sql_query("SELECT * FROM records ORDER BY id ASC", conn)

    if len(df) > 0:
        delete_id = st.selectbox("Select Record ID to Delete", df["id"])

        # Display selected record details
        selected_record = df[df["id"] == delete_id]
        st.subheader("🧾 Selected Record Details")
        st.dataframe(selected_record)

        if st.button("🗑️ Delete Selected Record"):
            c.execute("DELETE FROM records WHERE id = ?", (delete_id,))
            conn.commit()
            st.success(f"✅ Record with ID {delete_id} deleted successfully!")
    else:
        st.info("No records available to delete yet.")
