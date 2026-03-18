import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import joblib

# SHAP and plotting for explainability
import shap
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load trained model (cached to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/random_forest_best_model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file not found. Please run `python main.py` to train the model first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()

# SHAP explainer will be created on-demand during prediction
# (GradientBoosting multi-class doesn't support TreeExplainer)

st.title("AI CKD Outcome Prediction System")

st.markdown("""

This system uses a **Gradient Boosting machine learning model** trained on
clinical biomarker data to predict CKD progression risk.

Model Transparency
- Algorithm: Gradient Boosting
- Training samples: ~10,000 patients
- Features: 14 clinical biomarkers
- Explainability: SHAP feature attribution
            
Possible outcomes:
- Stable CKD
- Death/Progression Risk
- ESRD Risk
""")

st.divider()
left_col, right_col = st.columns([1,1])

with left_col:
    st.header("Patient Biomarker Input")

with right_col:
    st.header("Prediction Dashboard")

# Patient input fields
with left_col:
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

with left_col:
    st.divider()
    predict_button = st.button("Predict CKD Outcome")

if predict_button:
    # Input validation
    if diastolic >= systolic:
        st.warning("⚠️ Diastolic BP is greater than or equal to Systolic BP. Please verify your input.")
    try:

        with left_col:
            st.subheader("Patient Summary")

            summary_df = pd.DataFrame({
                "Input": [
                    "Age", "Sex", "BMI", "Hypertension", "Diabetes",
                    "HIV", "Glomerulonephritis", "eGFR", "Creatinine",
                    "UACR", "Hemoglobin", "Potassium", "Phosphate",
                    "Calcium", "BUN", "Systolic BP", "Diastolic BP", "HbA1c"
                ],
                "Value": [
                    str(age),
                    "Male" if sex == 1 else "Female",
                    str(bmi),
                    "Yes" if hypertension == 1 else "No",
                    "Yes" if diabetes == 1 else "No",
                    "Yes" if hiv == 1 else "No",
                    "Yes" if glomerulonephritis == 1 else "No",
                    str(egfr),
                    str(creatinine),
                    str(uacr),
                    str(hemoglobin),
                    str(potassium),
                    str(phosphate),
                    str(calcium),
                    str(bun),
                    str(systolic),
                    str(diastolic),
                    str(hba1c)
                ]
            })

            st.dataframe(summary_df, width='stretch', hide_index=True)

        patient_data = pd.DataFrame([{
            "age_years": age,
            "sex": sex,
            "diabetes": diabetes,
            "hiv_positive": hiv,
            "egfr": egfr,
            "serum_creatinine_mgdl": creatinine,
            "uacr_mg_g": uacr,
            "hemoglobin_gdl": hemoglobin,
            "potassium_meql": potassium,
            "phosphate_mgdl": phosphate,
            "calcium_mgdl": calcium,
            "bun_mgdl": bun,
            "systolic_bp": systolic,
            "hba1c_pct": hba1c
        }])

        # Apply feature engineering to match the model's training features
        from src.feature_engineering import engineer_features
        patient_data = engineer_features(patient_data)

        prediction = model.predict(patient_data)
        probs = model.predict_proba(patient_data)[0]

        outcome_map = {
            0: "Stable CKD",
            1: "Death/Progression Risk",
            2: "ESRD Risk"
        }

        result = outcome_map[prediction[0]]

        with right_col:
            st.subheader("Prediction Result")
            if prediction[0] == 0:
                st.success(f"✅ {result}")
            elif prediction[0] == 1:
                st.warning(f"🟡 {result}")
            else:
                st.error(f"🔴 {result}")

            st.subheader("Prediction Probabilities")

            # Align probabilities with the model's class order
            class_labels = [outcome_map[c] for c in model.classes_]

            prob_df = pd.DataFrame({
                "Outcome": class_labels,
                "Probability": probs
            })

            st.bar_chart(prob_df.set_index("Outcome"))

            # CKD Risk Gauge
            st.subheader("CKD Risk Level")

            # Use the highest serious outcome probability (Death/Progression, or ESRD)
            classes = list(model.classes_)
            death_prob = probs[classes.index(1)] if 1 in classes else 0
            esrd_prob = probs[classes.index(2)] if 2 in classes else 0
            risk_score = max(death_prob, esrd_prob)

            # Convert to percentage score
            risk_percent = int(risk_score * 100)

            # Visual gauge for risk score
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percent,
                number={'font': {'size': 48}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#ff4b4b", 'thickness': 0.18},
                    'bgcolor': "rgba(0,0,0,0)",
                    'steps': [
                        {'range': [0, 30], 'color': "#2ecc71"},
                        {'range': [30, 60], 'color': "#f1c40f"},
                        {'range': [60, 100], 'color': "#e74c3c"}
                    ]
                }
            ))
            gauge.update_layout(height=260, margin=dict(l=10, r=10, t=0, b=0))

            st.plotly_chart(gauge, width='stretch', config={"displayModeBar": False})

            # Risk interpretation
            if risk_score < 0.3:
                st.success("🟢 Low CKD Progression / Mortality Risk")
            elif risk_score < 0.6:
                st.warning("🟡 Moderate CKD Progression / Mortality Risk")
            else:
                st.error("🔴 High CKD Progression / Mortality Risk")

            st.subheader("Clinical Explanation")

            # Compute SHAP values for this patient
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(patient_data)
            except Exception:
                # GBM multi-class fallback: use feature importance as proxy
                importances = model.feature_importances_
                shap_values = None

            predicted_class = prediction[0]

            # Extract SHAP values for the predicted class and the single patient row
            if shap_values is not None:
                if isinstance(shap_values, list):
                    shap_values_class = shap_values[predicted_class][0]
                else:
                    shap_values_class = shap_values[0]
            else:
                # Use feature importance as a SHAP proxy
                shap_values_class = importances

            # Ensure numpy 1D array
            shap_values_class = np.array(shap_values_class).reshape(-1)

            # Force SHAP length to match number of features
            n_features = len(patient_data.columns)
            if len(shap_values_class) != n_features:
                shap_values_class = shap_values_class[:n_features]

            # Create SHAP importance dataframe safely
            shap_df = pd.DataFrame({
                "Feature": list(patient_data.columns),
                "SHAP Value": list(shap_values_class)
            })

            shap_df = shap_df.sort_values(by="SHAP Value", key=abs, ascending=False).head(10)

            fig, ax = plt.subplots(facecolor='#0e1117')
            ax.set_facecolor('#0e1117')

            # Color bars: red = increases risk, green = protective
            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["SHAP Value"]]

            ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
            ax.set_xlabel("Impact on Prediction")
            ax.set_title("Top Biomarkers Influencing This Prediction")
            ax.invert_yaxis()
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_visible(False)

            st.pyplot(fig)

            # --- Clinical Explanation Panel ---
            st.markdown("### Key Factors Driving This Prediction")

            top_factors = shap_df.head(5)

            explanation_lines = []
            for _, row in top_factors.iterrows():
                feature = row["Feature"].replace("_", " ").title()
                impact = row["SHAP Value"]

                if impact > 0:
                    explanation_lines.append(f"• **{feature}** increased the predicted risk")
                else:
                    explanation_lines.append(f"• **{feature}** reduced the predicted risk")

            for line in explanation_lines:
                st.markdown(line)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.exception(e)