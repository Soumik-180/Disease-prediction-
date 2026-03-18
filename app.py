import streamlit as st
st.set_page_config(layout="wide")
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import pandas as pd
import joblib

# SHAP and plotting for explainability
import shap
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load trained model (no caching to ensure latest version is used)
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

        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        hiv = st.selectbox("HIV Positive", ["No", "Yes"])
        
        bun = st.number_input("BUN (mg/dL)", 0.0, 100.0, 20.0)
        hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5)
        systolic = st.number_input("Systolic BP", 80, 250, 120)

    with col2:
        egfr = st.number_input("eGFR", 0.0, 200.0, 90.0)
        creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.0, 30.0, 1.0)
        uacr = st.number_input("UACR (mg/g)", 0.0, 5000.0, 50.0)

        hemoglobin = st.number_input("Hemoglobin (g/dL)", 0.0, 20.0, 13.0)
        potassium = st.number_input("Potassium (mEq/L)", 2.0, 7.0, 4.5)
        phosphate = st.number_input("Phosphate (mg/dL)", 1.0, 10.0, 4.0)
        calcium = st.number_input("Calcium (mg/dL)", 5.0, 12.0, 9.0)
        diastolic = st.number_input("Diastolic BP", 40, 150, 80)

# Convert categorical inputs
sex = 1 if sex == "Male" else 0
diabetes = 1 if diabetes == "Yes" else 0
hiv = 1 if hiv == "Yes" else 0

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
                "Feature": [
                    "Age", "Sex", "BMI", "Diabetes", "HIV Positive", 
                    "eGFR", "Creatinine", "UACR", "Hemoglobin", 
                    "Potassium", "Phosphate", "Calcium", "BUN", 
                    "Systolic BP", "Diastolic BP", "HbA1c"
                ],
                "Value": [
                    str(age),
                    "Male" if sex == 1 else "Female",
                    str(bmi),
                    "Yes" if diabetes == 1 else "No",
                    "Yes" if hiv == 1 else "No",
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
            "hba1c_pct": hba1c,
            # Add these if you ever collect them in the UI, else will be dropped below
            "bmi": bmi,
            "diastolic_bp": diastolic,
            # "hypertension": 0,  # Uncomment if you add to UI
            # "glomerulonephritis": 0,  # Uncomment if you add to UI
        }])

        # Drop the same features as in training before feature engineering
        features_to_drop = ['hypertension', 'glomerulonephritis', 'bmi', 'diastolic_bp']
        patient_data = patient_data.drop(columns=features_to_drop, errors='ignore')

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

            # Risk interpretation based on model prediction
            if prediction[0] == 0:
                st.success("🟢 Low Risk: Model predicts Stable CKD")
            elif prediction[0] == 1:
                st.warning("🟡 High Risk: Model predicts CKD Progression / Mortality")
            else:
                st.error("🔴 Severe Risk: Model predicts End-Stage Renal Disease (ESRD)")

            st.subheader("Clinical Explanation")

            # --- 1. Compute True Local SHAP Values ---
            with st.spinner("Analyzing personalized biomarker impacts..."):
                try:
                    # Load a small background dataset to serve as the baseline for SHAP
                    from config import CLEANED_DATA_PATH
                    df_bg = pd.read_csv(CLEANED_DATA_PATH)
                    target_col = 'outcome' if 'outcome' in df_bg.columns else df_bg.columns[-1]
                    X_bg = df_bg.drop(columns=[target_col], errors='ignore')
                    
                    # Apply the same feature engineering to the background
                    from src.feature_engineering import engineer_features
                    # Drop the 4 culled features from background just like we did for training
                    features_to_drop = ['hypertension', 'glomerulonephritis', 'bmi', 'diastolic_bp']
                    X_bg = X_bg.drop(columns=features_to_drop, errors='ignore')
                    X_bg = engineer_features(X_bg).sample(n=50, random_state=42)
                    
                    predicted_class = prediction[0]
                    
                    # We explain the predicted class probability
                    predict_fn = lambda x: model.predict_proba(x)[:, predicted_class]
                    explainer = shap.Explainer(predict_fn, X_bg)
                    shap_obj = explainer(patient_data)
                    
                    shap_values_class = shap_obj.values[0]
                    
                    # Reconstruct shap_df for the legacy bar chart if needed, though we will use waterfall
                    shap_df = pd.DataFrame({
                        "Feature": list(patient_data.columns),
                        "SHAP Value": list(shap_values_class)
                    }).sort_values(by="SHAP Value", key=abs, ascending=False)
                    
                except Exception as e:
                    st.error(f"SHAP Error: {e}")
                    # GBM/Pipeline multi-class fallback: use feature importance as proxy
                    try:
                        importances = model.named_steps['gb'].feature_importances_
                    except AttributeError:
                        importances = getattr(model, "feature_importances_", np.zeros(len(patient_data.columns)))
                    
                    shap_values_class = importances
                    # Ensure numpy 1D array
                    shap_values_class = np.array(shap_values_class).reshape(-1)
                    
                    if len(shap_values_class) != len(patient_data.columns):
                        shap_values_class = np.zeros(len(patient_data.columns))
                    
                    shap_df = pd.DataFrame({
                        "Feature": list(patient_data.columns),
                        "SHAP Value": list(shap_values_class)
                    }).sort_values(by="SHAP Value", key=abs, ascending=False)
                    
                    shap_obj = None

            # --- 2. SHAP Waterfall Visualization ---
            if shap_obj is not None:
                fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0e1117')
                ax.set_facecolor('#0e1117')
                
                # Plotly/Matplotlib styling for dark mode
                plt.rcParams.update({
                    "figure.facecolor": "#0e1117",
                    "axes.facecolor": "#0e1117",
                    "axes.edgecolor": "white",
                    "axes.labelcolor": "white",
                    "text.color": "white",
                    "xtick.color": "white",
                    "ytick.color": "white"
                })
                
                shap.plots.waterfall(shap_obj[0], show=False)
                
                # Force white text for standard SHAP plots which often override rcParams
                import matplotlib.text as mtext
                for ax_obj in fig.axes:
                    ax_obj.tick_params(colors="white")
                    ax_obj.xaxis.label.set_color("white")
                    ax_obj.yaxis.label.set_color("white")
                
                # Safely change all Text elements (labels, values, ticks) to white
                for text_obj in fig.findobj(match=mtext.Text):
                    text_obj.set_color("white")
                        
                st.pyplot(fig, transparent=True)
            else:
                st.warning("Could not generate SHAP Waterfall plot. Using fallback feature importances.")

            # --- 3. Enhanced Clinical Explanations ---
            st.markdown("### 🧬 Personalized Clinical Insights")
            
            clinical_context = {
                "bun_creatinine_ratio": "Elevated BUN/Cr ratios suggest dehydration or decreased kidney blood flow (pre-renal state), which acts as a major stressor on kidney function.",
                "calcium_phosphate_product": "High Ca-P product increases the risk of vascular calcification and cardiovascular events, which are highly correlated with progressive CKD.",
                "renal_risk_score": "This composite score aggregates age, blood pressure, and key kidney markers to estimate overall physiological stress.",
                "uacr_log": "Log UACR normalizes extreme proteinuria values. High proteinuria is one of the strongest predictors of rapid kidney function decline.",
                "egfr": "Estimated Glomerular Filtration Rate directly measures kidney filtration capacity. Lower values mean worse kidney function.",
                "serum_creatinine_mgdl": "Creatinine is a waste product. Elevated levels indicate the kidneys are currently failing to filter waste from the blood.",
                "age_years": "Older age naturally correlates with reduced kidney function elasticity and higher susceptibility to progression.",
                "hemoglobin_gdl": "Low hemoglobin (anemia) is a common complication of advanced CKD due to decreased erythropoietin production in the kidneys.",
                "systolic_bp": "High blood pressure accelerates kidney damage by stressing the delicate filtration blood vessels (glomeruli).",
                "diabetes": "Diabetes is a leading cause of CKD, causing diabetic nephropathy and structural kidney damage over time.",
                "potassium_meql": "Failing kidneys cannot excrete potassium properly, leading to life-threatening hyperkalemia.",
                "phosphate_mgdl": "High phosphate levels occur in advanced CKD and can cause severe bone and heart problems.",
                "calcium_mgdl": "Abnormal calcium levels are linked to CKD mineral and bone disorder.",
                "bun_mgdl": "Blood Urea Nitrogen buildup indicates poor waste filtration by the kidneys.",
                "hba1c_pct": "High HbA1c indicates poor long-term blood sugar control, accelerating diabetic kidney damage.",
                "sex": "Biological sex can influence CKD progression rates and baseline creatinine distribution.",
                "hiv_positive": "HIV infection or its treatments can cause HIV-associated nephropathy (HIVAN), increasing CKD risk."
            }

            top_factors = shap_df.head(4)  # Show top 4 drivers

            for _, row in top_factors.iterrows():
                feat_name = row["Feature"]
                impact = row["SHAP Value"]
                
                # Format the feature name for reading
                clean_name = feat_name.replace("_", " ").title()
                
                # Get the actual patient value for context
                val = patient_data[feat_name].iloc[0]
                if isinstance(val, float):
                    val = round(val, 2)
                
                # Context message
                context_msg = clinical_context.get(feat_name, "Significant contributing biomarker.")
                
                if impact > 0:
                    st.error(f"📈 **Elevated Risk from {clean_name} (Value: {val})**")
                    st.caption(f"{context_msg}")
                else:
                    st.success(f"📉 **Protective/Stable Factor: {clean_name} (Value: {val})**")
                    st.caption(f"{context_msg}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.exception(e)