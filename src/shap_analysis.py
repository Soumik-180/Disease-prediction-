import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


def run_shap_analysis():
    print("\nRunning SHAP Explainability Analysis")

    # Paths from centralized config
    from config import MODEL_PATH, SPLIT_DIR, FIGURES_DIR
    model_path = MODEL_PATH
    test_data_path = f"{SPLIT_DIR}/X_test.csv"
    figures_dir = FIGURES_DIR

    os.makedirs(figures_dir, exist_ok=True)

    # Load trained model
    model = joblib.load(model_path)
    print("Model loaded successfully")

    # Load test dataset
    X_test = pd.read_csv(test_data_path)
    print("Test dataset loaded")

    # Use a subset for faster SHAP computation
    X_sample = X_test.sample(n=min(100, len(X_test)), random_state=42)
    
    features_to_drop = ['hypertension', 'glomerulonephritis', 'bmi', 'diastolic_bp']
    X_sample = X_sample.drop(columns=features_to_drop, errors='ignore')

    # Apply feature engineering to match the model's training features
    from src.feature_engineering import engineer_features
    X_sample = engineer_features(X_sample)

    # Create SHAP explainer — use TreeExplainer if supported, else fall back
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample, check_additivity=False)
    except Exception as e:
        print(f"TreeExplainer not supported ({e}), using Explainer fallback...")
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample).values

    print("SHAP values computed")

    # -------- SHAP Summary Plot --------
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )

    summary_path = os.path.join(figures_dir, "shap_summary_plot.png")
    plt.savefig(summary_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"SHAP summary plot saved at: {summary_path}")

    # -------- SHAP Feature Importance Plot --------
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )

    bar_path = os.path.join(figures_dir, "shap_feature_importance.png")
    plt.savefig(bar_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"SHAP feature importance plot saved at: {bar_path}")

    print("\nSHAP Analysis Completed")


if __name__ == "__main__":
    run_shap_analysis()