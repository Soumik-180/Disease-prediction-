#!/usr/bin/env python3
"""
CKD Dataset Train/Test Split Script

Purpose:
Split the cleaned CKD dataset into training and testing sets
while preserving class distribution.

Best Practices Used:
- Stratified splitting
- Reproducibility with random_state
- Saving datasets for reproducibility
- Verifying class balance

Author: ML Engineering Pipeline
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from config import CLEANED_DATA_PATH, SPLIT_DIR

# Paths
INPUT_PATH = CLEANED_DATA_PATH
OUTPUT_DIR = SPLIT_DIR

# Create split directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_dataset():

    print("=== Loading Cleaned Dataset ===")

    # Load dataset
    df = pd.read_csv(INPUT_PATH)

    # --- Apply outcome class merging to match training pipeline ---
    # --- Target class labels are merged to 3 classes ---
    # Original labels: 0=Stable, 1=Progression, 2=Death, 3=ESRD
    df["outcome"] = df["outcome"].astype(int).map({0: 0, 1: 1, 2: 1, 3: 2})

    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    print("\nOutcome Distribution After Class Merging:")
    print(df["outcome"].value_counts().sort_index())
    print(df["outcome"].value_counts(normalize=True).sort_index())

    # Separate features and target
    X = df.drop(columns=["outcome"])
    y = df["outcome"]

    print("\n=== Splitting Dataset (80/20) ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,          # Keeps class proportions
        random_state=42
    )

    # Reset indices to keep saved CSV files clean
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("\nTraining Set Shape:", X_train.shape)
    print("Test Set Shape:", X_test.shape)

    print("\nTraining Outcome Distribution:")
    print(y_train.value_counts().sort_index())
    print(y_train.value_counts(normalize=True).sort_index())

    print("\nTest Outcome Distribution:")
    print(y_test.value_counts().sort_index())
    print(y_test.value_counts(normalize=True).sort_index())

    # Save split datasets
    print("\n=== Saving Split Datasets ===")

    X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

    print("Saved files:")
    print(f"{OUTPUT_DIR}/X_train.csv")
    print(f"{OUTPUT_DIR}/X_test.csv")
    print(f"{OUTPUT_DIR}/y_train.csv")
    print(f"{OUTPUT_DIR}/y_test.csv")

    print("\nDataset split completed successfully.")

if __name__ == "__main__":
    split_dataset()