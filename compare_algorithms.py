"""
Multi-Algorithm Comparison Script for CKD Outcome Prediction

Trains 6 different algorithms on the same data and compares their performance.
Uses the same 14-feature culled dataset and SMOTEENN resampling.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, balanced_accuracy_score,
                             make_scorer)
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---- Load Data ----
print("=" * 70)
print("MULTI-ALGORITHM COMPARISON FOR CKD OUTCOME PREDICTION")
print("=" * 70)

X_train = pd.read_csv('Data/Split/X_train.csv')
X_test = pd.read_csv('Data/Split/X_test.csv')
y_train = pd.read_csv('Data/Split/y_train.csv').squeeze()
y_test = pd.read_csv('Data/Split/y_test.csv').squeeze()

# Apply same feature culling
features_to_drop = ['hypertension', 'glomerulonephritis', 'bmi', 'diastolic_bp']
X_train = X_train.drop(columns=features_to_drop, errors='ignore')
X_test = X_test.drop(columns=features_to_drop, errors='ignore')

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Features: {list(X_train.columns)}")
print(f"\nClass distribution (train):")
print(y_train.value_counts().sort_index())

# ---- Define Models ----
models = {
    "Random Forest": ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('clf', RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_split=10,
            min_samples_leaf=4, max_features='sqrt',
            class_weight='balanced', random_state=42
        ))
    ]),

    "XGBoost": ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('clf', XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=42,
            use_label_encoder=False
        ))
    ]),

    "LightGBM": ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('clf', LGBMClassifier(
            n_estimators=400, max_depth=-1, learning_rate=0.1,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, verbose=-1
        ))
    ]),

    "Gradient Boosting": ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('clf', GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, max_features='sqrt', random_state=42
        ))
    ]),

    "SVM (RBF)": ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('clf', SVC(
            kernel='rbf', C=10, gamma='scale',
            class_weight='balanced', probability=True, random_state=42
        ))
    ]),

    "Logistic Regression": ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('clf', LogisticRegression(
            max_iter=1000, class_weight='balanced',
            solver='lbfgs', random_state=42
        ))
    ]),
}

# ---- Class labels ----
class_names = ["Stable CKD", "Death/Progression", "ESRD"]

# ---- Train and Evaluate ----
results = []

for name, pipeline in models.items():
    print(f"\n{'─' * 70}")
    print(f"Training: {name}")
    print(f"{'─' * 70}")

    # Fit
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=[0, 1, 2],
                                   target_names=class_names, output_dict=True)

    stable_f1 = report["Stable CKD"]["f1-score"]
    death_recall = report["Death/Progression"]["recall"]
    death_precision = report["Death/Progression"]["precision"]
    death_f1 = report["Death/Progression"]["f1-score"]
    esrd_recall = report["ESRD"]["recall"]
    esrd_f1 = report["ESRD"]["f1-score"]
    macro_f1 = report["macro avg"]["f1-score"]
    overall_acc = report["accuracy"]

    results.append({
        "Algorithm": name,
        "Balanced Acc": f"{bal_acc:.4f}",
        "Overall Acc": f"{overall_acc:.4f}",
        "Macro F1": f"{macro_f1:.4f}",
        "Stable F1": f"{stable_f1:.4f}",
        "Death Recall": f"{death_recall:.4f}",
        "Death Prec": f"{death_precision:.4f}",
        "Death F1": f"{death_f1:.4f}",
        "ESRD Recall": f"{esrd_recall:.4f}",
        "ESRD F1": f"{esrd_f1:.4f}",
    })

    # Print classification report
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(classification_report(y_test, y_pred, labels=[0, 1, 2],
                                target_names=class_names))

# ---- Comparison Table ----
print("\n" + "=" * 70)
print("FINAL COMPARISON TABLE")
print("=" * 70)

results_df = pd.DataFrame(results)
# Sort by Balanced Accuracy descending
results_df = results_df.sort_values("Balanced Acc", ascending=False)
print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv("results/algorithm_comparison.csv", index=False)
print("\nComparison saved to results/algorithm_comparison.csv")

# ---- Winner ----
best = results_df.iloc[0]
print(f"\n🏆 BEST ALGORITHM: {best['Algorithm']}")
print(f"   Balanced Accuracy: {best['Balanced Acc']}")
print(f"   Death/Progression Recall: {best['Death Recall']}")
print(f"   ESRD Recall: {best['ESRD Recall']}")
