import os
import joblib
import pandas as pd
from config import CLEANED_DATA_PATH, SPLIT_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR, MODEL_PATH
from src.data_cleaning import run_cleaning_v3
from src.data_split import split_dataset
from src.train_model import perform_training
from src.evaluate_model import evaluate_and_plot
from src.feature_importance import analyze_feature_importance
from src.shap_analysis import run_shap_analysis


def main():
    # Configuration (paths from config.py)
    cleaned_file = CLEANED_DATA_PATH
    split_dir = SPLIT_DIR
    models_dir = MODELS_DIR
    figures_dir = FIGURES_DIR
    results_dir = RESULTS_DIR

    # Ensure directories exist
    for d in [models_dir, figures_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    print("\n" + "=" * 50)
    print("CKD Biomarker Discovery Pipeline")
    print("=" * 50)

    # Step 1 — Data Cleaning (skip if cleaned file already exists)
    if not os.path.exists(cleaned_file):
        print("\nCleaned dataset not found. Running data cleaning...")
        run_cleaning_v3()
    else:
        print(f"\nCleaned dataset found at {cleaned_file}. Skipping cleaning.")

    # Step 2 — Data Splitting (skip if split files already exist)
    split_files = [f'{split_dir}/X_train.csv', f'{split_dir}/X_test.csv',
                   f'{split_dir}/y_train.csv', f'{split_dir}/y_test.csv']
    if not all(os.path.exists(f) for f in split_files):
        print("\nSplit files not found. Running data splitting...")
        split_dataset()
    else:
        print(f"\nSplit files found in {split_dir}/. Skipping splitting.")

    # Step 3 — Load train/test splits
    X_train = pd.read_csv(f'{split_dir}/X_train.csv')
    X_test = pd.read_csv(f'{split_dir}/X_test.csv')
    y_train = pd.read_csv(f'{split_dir}/y_train.csv').squeeze()
    y_test = pd.read_csv(f'{split_dir}/y_test.csv').squeeze()

    # Step 4 — Train model (returns imblearn Pipeline with SMOTE + RF)
    best_pipeline = perform_training(X_train, X_test, y_train, y_test)

    # Extract the trained RF model from the pipeline
    # (needed for feature importance and saving a clean model for app.py)
    best_model = best_pipeline.named_steps['rf']

    # Step 5 — Evaluate model (pipeline handles prediction correctly)
    evaluate_and_plot(best_pipeline, X_test, y_test, figures_dir=figures_dir)

    # Step 6 — Feature importance (uses the raw RF model)
    analyze_feature_importance(
        best_model,
        feature_names=X_train.columns,
        results_dir=results_dir,
        figures_dir=figures_dir
    )

    # Step 8 — SHAP explainability analysis
    run_shap_analysis()

    # Step 9 — Save the RF model (not the full pipeline, for app.py compatibility)
    joblib.dump(best_model, MODEL_PATH)

    print("\nPipeline Execution Successful")


if __name__ == "__main__":
    main()
