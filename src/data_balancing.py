import pandas as pd

def balance_outcome_classes(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Merges 'Progression' (1) and 'Death' (2) into a single 'Death' class
    to address class imbalance.

    New Class Labels:
        0 -> Stable CKD
        1 -> Death (merged from Progression + Death)
        2 -> ESRD

    Args:
        X: Feature DataFrame.
        y: Target Series with original outcome labels.

    Returns:
        Tuple of (X unchanged, y with merged class labels).
    """
    print("\n=== Step 2: Fixing Outcome Imbalance ===")
    
    # Create a copy to avoid SettingWithCopyWarning
    y_balanced = y.copy()
    
    # Merge classes only if the dataset still contains the original 4 labels
    unique_labels = set(y_balanced.unique())

    if max(unique_labels) > 2:
        # Original dataset with labels 0,1,2,3
        y_balanced = y_balanced.map({
            0: 0,  # Stable CKD
            1: 1,  # Death
            2: 1,  # Death (Merged)
            3: 2   # ESRD
        }).astype(int)
    else:
        # Dataset already merged (0,1,2) — keep as is
        y_balanced = y_balanced.astype(int)
    
    print("New Class Distribution (Merged):")
    class_mapping = {
        0: "Stable CKD",
        1: "Death",
        2: "ESRD"
    }
    counts = y_balanced.value_counts().sort_index()
    percentages = y_balanced.value_counts(normalize=True).sort_index()

    for label, count in counts.items():
        pct = percentages[label] * 100
        print(f"  {class_mapping[label]} ({label}): {count} ({pct:.2f}%)")
    
    return X, y_balanced
