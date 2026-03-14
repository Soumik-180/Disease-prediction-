import os
import joblib
import pandas as pd
from src.data_loader import load_dataset
from src.data_balancing import balance_outcome_classes
from src.train_model import perform_training
from src.evaluate_model import evaluate_and_plot
from src.feature_importance import analyze_feature_importance


def main():
    # Configuration
    input_file = 'Data/Cleaned/ckd_cleaned_dataset_v3.csv'
    models_dir = 'models'
    figures_dir = 'figures'
    results_dir = 'results'

    # Ensure directories exist
    for d in [models_dir, figures_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    print("\n" + "=" * 50)
    print("CKD Biomarker Discovery Pipeline")
    print("=" * 50)

    # Step 1 — Load cleaned dataset
    X, y = load_dataset(input_file)

    # Step 2 — Merge outcome classes
    X, y = balance_outcome_classes(X, y)

    # Step 3 — Load train/test splits
    X_train = pd.read_csv('Data/Split/X_train.csv')
    X_test = pd.read_csv('Data/Split/X_test.csv')
    y_train = pd.read_csv('Data/Split/y_train.csv').squeeze()
    y_test = pd.read_csv('Data/Split/y_test.csv').squeeze()

    # Ensure class merging applied to splits
    X_train, y_train = balance_outcome_classes(X_train, y_train)
    X_test, y_test = balance_outcome_classes(X_test, y_test)

    # Step 4 — Train model
    best_model = perform_training(X_train, X_test, y_train, y_test)

    # Step 5 — Evaluate model
    evaluate_and_plot(best_model, X_test, y_test, figures_dir=figures_dir)

    # Step 6 — Feature importance
    analyze_feature_importance(
        best_model,
        feature_names=X.columns,
        results_dir=results_dir,
        figures_dir=figures_dir
    )

    # Step 7 — Save trained model
    model_path = os.path.join(models_dir, 'random_forest_best_model.pkl')
    joblib.dump(best_model, model_path)

    print("\nPipeline Execution Successful")


if __name__ == "__main__":
    main()
