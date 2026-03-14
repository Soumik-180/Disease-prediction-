import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/random_forest_best_model.pkl")

st.title("AI CKD Outcome Prediction System")

st.write("Enter patient biomarker values to predict CKD outcome.")

st.markdown("""
### AI Biomarker-Based CKD Outcome Prediction

This system uses a **Random Forest machine learning model** trained on
clinical biomarker data to predict CKD progression risk.

Possible outcomes:
- Stable CKD
- Death Risk
- ESRD Risk
""")

# Patient input fields (organized in columns)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", 18, 100, 50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)

    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    hiv = st.selectbox("HIV Positive", ["No", "Yes"])
    glomerulonephritis = st.selectbox("Glomerulonephritis", ["No", "Yes"])

with col2:
    egfr = st.number_input("eGFR", 0.0, 200.0, 90.0)
    creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.0, 30.0, 1.0)
    uacr = st.number_input("UACR (mg/g)", 0.0, 5000.0, 50.0)

    hemoglobin = st.number_input("Hemoglobin (g/dL)", 0.0, 20.0, 13.0)
    potassium = st.number_input("Potassium (mEq/L)", 2.0, 7.0, 4.5)
    phosphate = st.number_input("Phosphate (mg/dL)", 1.0, 10.0, 4.0)
    calcium = st.number_input("Calcium (mg/dL)", 5.0, 12.0, 9.0)

bun = st.number_input("BUN (mg/dL)", 0.0, 100.0, 20.0)
systolic = st.number_input("Systolic BP", 80, 250, 120)
diastolic = st.number_input("Diastolic BP", 40, 150, 80)
hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5)

# Convert categorical inputs
sex = 1 if sex == "Male" else 0
hypertension = 1 if hypertension == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
hiv = 1 if hiv == "Yes" else 0
glomerulonephritis = 1 if glomerulonephritis == "Yes" else 0

# Prediction button
if st.button("Predict CKD Outcome"):

    patient_data = pd.DataFrame([{
        "age_years": age,
        "sex": sex,
        "bmi": bmi,
        "hypertension": hypertension,
        "diabetes": diabetes,
        "hiv_positive": hiv,
        "glomerulonephritis": glomerulonephritis,
        "egfr": egfr,
        "serum_creatinine_mgdl": creatinine,
        "uacr_mg_g": uacr,
        "hemoglobin_gdl": hemoglobin,
        "potassium_meql": potassium,
        "phosphate_mgdl": phosphate,
        "calcium_mgdl": calcium,
        "bun_mgdl": bun,
        "systolic_bp": systolic,
        "diastolic_bp": diastolic,
        "hba1c_pct": hba1c
    }])

    prediction = model.predict(patient_data)
    probs = model.predict_proba(patient_data)[0]

    outcome_map = {
        0: "Stable CKD",
        1: "Death Risk",
        2: "ESRD Risk"
    }

    result = outcome_map[prediction[0]]

    st.subheader("Prediction Result")
    st.success(result)

    st.subheader("Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Outcome": ["Stable CKD", "Death", "ESRD"],
        "Probability": probs
    })

    st.bar_chart(prob_df.set_index("Outcome"))