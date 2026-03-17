from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd

def perform_training(X_train, X_test, y_train, y_test, random_state=42):
    """
    Train model using an imblearn Pipeline to apply SMOTE within each CV fold,
    preventing data leakage. Performs hyperparameter tuning via RandomizedSearchCV.
    """
    print("\n=== Step 3: Model Training & Hyperparameter Tuning ===")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Build pipeline: SMOTE is applied inside each CV fold (no leakage)
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('rf', RandomForestClassifier(random_state=random_state, class_weight='balanced'))
    ])

    # Hyperparameter grid (prefixed with 'rf__' for the pipeline step)
    param_dist = {
        'rf__n_estimators': [300, 400, 500, 600],
        'rf__max_depth': [None, 10, 20, 30],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_features': ['sqrt', 'log2', None]
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="f1_weighted",
        random_state=random_state,
        n_jobs=-1
    )

    print("Performing RandomizedSearchCV (5-fold CV, F1-weighted)...")
    print("(SMOTE applied within each CV fold to prevent data leakage)")
    search.fit(X_train, y_train)

    print("\nBest Hyperparameters Found:")
    print(search.best_params_)
    print(f"Best CV F1 Score (weighted): {search.best_score_:.4f}")

    # Return the best pipeline (includes SMOTE + trained RF)
    best_model = search.best_estimator_

    return best_model