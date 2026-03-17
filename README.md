

# AI-Driven Biomarker-Based Disease Outcome Prediction System

## Overview

This project presents an AI-powered clinical decision-support system designed to analyze biomarker patterns and predict disease outcomes. The system uses machine learning and explainable AI to assist in early risk detection and improve clinical decision-making.

The current implementation demonstrates a prototype using Chronic Kidney Disease (CKD) as a use case, but the framework is designed to be extensible to multiple diseases.

---

## Key Features

- Multi-biomarker analysis
- AI-based outcome prediction
- Explainable AI (SHAP)
- Risk visualization dashboard
- Clinical interpretation of results
- Interactive web interface (Streamlit)
- Deployable cloud application

---

## Dataset

The dataset consists of clinical biomarker features such as:

- Age
- Sex
- BMI
- Hypertension
- Diabetes
- eGFR
- Serum Creatinine
- UACR
- Hemoglobin
- Potassium
- Phosphate
- Calcium
- BUN
- Blood Pressure
- HbA1c

Target variable:

- 0 → Stable CKD
- 1 → Death Risk
- 2 → ESRD

---

## Machine Learning Pipeline

1. Data Preprocessing
   - Missing value handling
   - Outlier removal
   - Encoding categorical variables

2. Class Imbalance Handling
   - SMOTE (Synthetic Minority Oversampling Technique)

3. Data Splitting
   - Train/Test split (80/20)

4. Model Training
   - Random Forest Classifier
   - Hyperparameter tuning using RandomizedSearchCV

5. Model Evaluation
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix

6. Explainability
   - SHAP (feature contribution analysis)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Soumik-180/Disease-prediction-.git
cd Disease-prediction-
```

Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run training pipeline:

```bash
python main.py
```

Run Streamlit app:

```bash
streamlit run app.py
```

---

## Deployment

The application is deployed using GitHub and Streamlit Community Cloud.

**Try the prototype:** [Live App Link](https://disease-outcome-prediction.streamlit.app) *(Update this link to your actual deployment)*

### How to deploy your own instance

1. **Push your code to GitHub:** Ensure your repository includes `app.py`, `requirements.txt`, and the trained `models/` folder. (Note: You may need Git LFS for large `.pkl` files).
2. **Go to Streamlit Community Cloud:** Log in at [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. **Deploy the app:** Click "New app", select your repository, branch, and set the Main file path to `app.py`. Click "Deploy".

---

## Results

- Balanced Accuracy achieved: ~0.61
- Model successfully predicts CKD outcomes
- SHAP explains biomarker contributions

---

## Future Scope

- Extend to multiple diseases
- Integrate real clinical datasets
- Use longitudinal patient data
- Add multi-omics analysis
- Integrate with hospital systems (EHR)

---

## Limitations

- Dataset is limited to biomarker features
- Mortality prediction requires additional clinical data
- External validation not yet performed

---

## Author

Soumik Ray
B.Tech Biotechnology (Computational Biology)
Sharda University

---

## License

This project is for academic and research purposes.