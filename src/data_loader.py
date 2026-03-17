import pandas as pd
import os

def load_dataset(file_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads the CKD dataset, prints overview statistics, and returns features/target.

    Args:
        file_path: Path to the cleaned CSV dataset.

    Returns:
        Tuple of (X features DataFrame, y target Series).
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path)

    print("\n=== Step 1: Data Loading & Exploration ===")

    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

    # Remove rows with missing outcome just in case
    df = df.dropna(subset=["outcome"])

    print("\nSummary Statistics (Numerical):")
    print(df.describe().T)

    print("\nInitial Outcome Class Counts:")
    print(df["outcome"].value_counts().sort_index())

    print("\nInitial Outcome Distribution (%):")
    print(df["outcome"].value_counts(normalize=True).sort_index())

    # Separate features and target
    X = df.drop(columns=["outcome"])
    y = df["outcome"]

    return X, y