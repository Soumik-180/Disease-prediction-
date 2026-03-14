from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd

def perform_training(X_train, X_test, y_train, y_test, random_state=42):
    """
    Split data, perform hyperparameter tuning, and train final model.
    """
    print("\n=== Step 3: Model Training & Hyperparameter Tuning ===")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # --- Step 8: Handle class imbalance using SMOTE (training data only) ---
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_train).value_counts().sort_index())
    
    # 2. RandomizedSearchCV (Hyperparameter Tuning)
    param_dist = {
        'n_estimators': [300, 400, 500, 600],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Random Forest with Balanced Weights
    rf = RandomForestClassifier(random_state=random_state)
    
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="f1_weighted",
        random_state=random_state,
        n_jobs=-1
    )
    
    print("Performing RandomizedSearchCV (5-fold CV, F1-weighted)...")
    search.fit(X_train, y_train)
    
    print("\nBest Hyperparameters Found:")
    print(search.best_params_)
    
# 3. Final Model
    best_model = search.best_estimator_

    return best_model