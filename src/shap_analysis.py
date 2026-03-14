import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


def run_shap_analysis():
    print("\nRunning SHAP Explainability Analysis")

    # Paths
    model_path = "models/random_forest_best_model.pkl"
    test_data_path = "Data/Split/X_test.csv"
    figures_dir = "figures"

    os.makedirs(figures_dir, exist_ok=True)

    # Load trained model
    model = joblib.load(model_path)
    print("Model loaded successfully")

    # Load test dataset
    X_test = pd.read_csv(test_data_path)
    print("Test dataset loaded")

    # Use a subset for faster SHAP computation
    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)

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