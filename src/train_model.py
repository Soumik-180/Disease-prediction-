from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from src.feature_engineering import engineer_features
import pandas as pd

def perform_training(X_train, X_test, y_train, y_test, random_state=42):
    """
    Train model using an imblearn Pipeline to apply SMOTEENN within each CV fold,
    preventing data leakage. Performs hyperparameter tuning via RandomizedSearchCV.
    Uses Gradient Boosting Classifier (winner of multi-algorithm comparison).
    """
    print("\n=== Step 3: Model Training & Hyperparameter Tuning ===")
    print(f"Original Training set size: {X_train.shape}")
    print(f"Original Test set size: {X_test.shape}")

    # Feature Culling: drop low-importance features
    features_to_drop = ['hypertension', 'glomerulonephritis', 'bmi', 'diastolic_bp']
    X_train = X_train.drop(columns=features_to_drop, errors='ignore')
    X_test = X_test.drop(columns=features_to_drop, errors='ignore')

    # Feature Engineering: add clinically meaningful derived features
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    print(f"Training set size after culling + engineering: {X_train.shape}")
    print(f"Test set size after culling + engineering: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")

    # Build pipeline: SMOTEENN is applied inside each CV fold (no leakage)
    pipeline = ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=random_state)),
        ('gb', GradientBoostingClassifier(random_state=random_state))
    ])

    # Hyperparameter grid (prefixed with 'gb__' for the pipeline step)
    param_dist = {
        'gb__n_estimators': [200, 300, 400, 500],
        'gb__max_depth': [3, 5, 7, 10],
        'gb__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'gb__subsample': [0.7, 0.8, 0.9, 1.0],
        'gb__min_samples_split': [2, 5, 10],
        'gb__min_samples_leaf': [1, 2, 4],
        'gb__max_features': ['sqrt', 'log2']
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=30,
        cv=5,
        scoring="f1_macro",
        random_state=random_state,
        n_jobs=-1
    )

    print("Performing RandomizedSearchCV (5-fold CV, F1-macro)...")
    print("(SMOTEENN applied within each CV fold to prevent data leakage)")
    search.fit(X_train, y_train)

    print("\nBest Hyperparameters Found:")
    print(search.best_params_)
    print(f"Best CV F1 Score (macro): {search.best_score_:.4f}")

    # Return the best pipeline (includes SMOTEENN + trained GB)
    best_model = search.best_estimator_

    return best_model