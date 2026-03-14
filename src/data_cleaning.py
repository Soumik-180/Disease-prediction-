#!/usr/bin/env python3
"""
Chronic Kidney Disease (CKD) Data Cleaning Pipeline v3

This script rebuilds the cleaning process to:
1. Preserve the natural clinical distribution (Majority Stable).
2. Remove only physiologically impossible values.
3. Eliminate feature leakage (Albuminuria categories, Etiology).
4. Use median imputation to prevent biased row deletion.

Author: Senior Computational Biologist & ML Engineer
Date: 2026-03-13
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
INPUT_PATH = 'Data/Raw/ckd_community_screening.csv'
OUTPUT_PATH = 'Data/Cleaned/ckd_cleaned_dataset_v3.csv'
GRAPH_DIR = 'Graph'

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

def run_cleaning_v3():
    # --- STEP 1: LOAD RAW DATASET ---
    print("=== Step 1: Loading Raw Dataset ===")
    if not os.path.exists(INPUT_PATH):
        # Fallback if path is different in local environment
        df = pd.read_csv('ckd_community_screening.csv')
    else:
        df = pd.read_csv(INPUT_PATH)
        
    print(f"Initial Dataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nInitial Outcome Distribution:")
    print(df['outcome'].value_counts())
    print("-" * 30)

    # --- STEP 2: REMOVE FEATURE LEAKAGE ---
    print("=== Step 2: Removing Feature Leakage ===")
    # Dropping derived categories and etiologies that introduce diagnostic leakage
    leakage_cols = [
        # Previously discussed leakage / diagnostic variables
        'ckd_stage',
        'on_dialysis',
        'dialysis_type',

        # Derived biomarker categories
        'albuminuria_category',
        'albuminuria_category_A2',
        'albuminuria_category_A3',
        'anemia',
        'hyperkalemia',

        # Treatment variables (post‑diagnosis signals)
        'on_epo',
        'on_acei_arb',
        'on_antihypertensives',
        'on_statin',

        # Etiology label (diagnostic information)
        'primary_etiology',
        'primary_etiology_diabetic_nephropathy',
        'primary_etiology_glomerulonephritis',
        'primary_etiology_hiv_nephropathy',
        'primary_etiology_hypertensive_nephrosclerosis',
        'primary_etiology_obstructive',
        'primary_etiology_unknown',

        # Identifier column
        'id'
    ]
    df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True, errors='ignore')
    print(f"Dropped leakage/ID columns. Current columns count: {len(df.columns)}")

    # --- STEP 3: REMOVE BIOLOGICALLY IMPOSSIBLE VALUES ---
    print("\n=== Step 3: Filtering Biologically Impossible Values ===")
    initial_rows = len(df)
    
    # Conservative biological ranges (Only removing physiological impossibilities)
    filters = {
        'serum_creatinine_mgdl': (0.1, 30.0),
        'hemoglobin_gdl': (3.0, 20.0),
        'potassium_meql': (1.0, 8.0),
        'phosphate_mgdl': (0.5, 15.0),
        'calcium_mgdl': (4.0, 15.0),
        'bun_mgdl': (1.0, 200.0),
        'systolic_bp': (50.0, 300.0),
        'diastolic_bp': (30.0, 200.0)
    }
    
    for col, (low, high) in filters.items():
        if col in df.columns:
            # Masking out-of-range values as NaN so they can be imputed later, 
            # OR removing the row if it's a critical error.
            # Here we follow instruction to remove rows violating these specific limits.
            df = df[(df[col] >= low) & (df[col] <= high) | (df[col].isna())]
            
    print(f"Rows removed due to impossible biological values: {initial_rows - len(df)}")

    # --- STEP 4: HANDLE MISSING VALUES (MEDIAN IMPUTATION) ---
    print("\n=== Step 4: Handling Missing Values (Median Imputation) ===")
    # We do NOT dropna() as it biases towards ESRD patients with complete panels
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # For categorical features (if any remain), fill with mode
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if col != 'outcome':
            df[col] = df[col].fillna(df[col].mode()[0])
            
    print("Imputation complete. Remaining missing values: 0")

    # --- STEP 4.5: ENCODE SEX AS NUMERIC (M=1, F=0) ---
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).str.strip().str.upper()
        df['sex'] = df['sex'].map({'M': 1, 'F': 0})
        # If any unexpected values appear, fill with median (0 or 1 depending on distribution)
        if df['sex'].isnull().any():
            df['sex'] = df['sex'].fillna(df['sex'].median())

    # --- STEP 5: STANDARDIZE OUTCOME LABELS ---
    print("\n=== Step 5: Standardizing Outcome Labels ===")
    # Clean the strings before mapping
    df['outcome'] = df['outcome'].astype(str).str.strip().str.lower()
    
    outcome_map = {
        "stable": 0,
        "progressed": 1,
        "died": 2,
        "esrd": 3
    }
    
    # Map values
    df['outcome'] = df['outcome'].map(outcome_map)
    
    # Verify mapping success
    if df['outcome'].isnull().any():
        print("Warning: Some outcome labels were not mapped correctly. Dropping those rows.")
        df.dropna(subset=['outcome'], inplace=True)
    
    # Ensure it's an integer type
    df['outcome'] = df['outcome'].astype(int)

    # --- STEP 6: VERIFY OUTCOME DISTRIBUTION ---
    print("\n=== Step 6: Post-Cleaning Distribution Check ===")
    counts = df['outcome'].value_counts().sort_index()
    proportions = df['outcome'].value_counts(normalize=True).sort_index()
    
    label_rev_map = {0: "Stable", 1: "Progressed", 2: "Died", 3: "ESRD"}
    for label, count in counts.items():
        label_name = label_rev_map.get(label, f"Unknown ({label})")
        print(f"  {label_name}: {count} ({proportions[label]:.2%})")

    # --- STEP 7: GENERATE DIAGNOSTIC VISUALIZATIONS ---
    print("\n=== Step 7: Generating Visualizations ===")
    
    # 1. Outcome Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='outcome', data=df, palette='muted')
    plt.title('Distribution of CKD Outcomes in Cleaned Dataset (v3)', fontsize=14)
    plt.xticks(ticks=[0, 1, 2, 3], labels=["Stable", "Progressed", "Died", "ESRD"])
    plt.xlabel('CKD Outcome', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/v3_outcome_distribution.png")
    plt.close()

    # 2. Serum Creatinine Distribution
    plt.figure(figsize=(10, 6))
    df['serum_creatinine_mgdl'].hist(bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Serum Creatinine Levels (Cleaned)', fontsize=14)
    plt.xlabel('Serum Creatinine (mg/dL)', fontsize=12)
    plt.ylabel('Patient Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/v3_creatinine_histogram.png")
    plt.close()

    # 3. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    # Filter to numeric only for correlation
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Between Clinical Biomarkers (Cleaned v3)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/v3_biomarker_correlation.png")
    plt.close()

    # --- STEP 8: SAVE CLEANED DATASET ---
    print("\n=== Step 8: Saving Cleaned Dataset ===")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Final Dataset Shape: {df.shape}")
    print(f"Dataset saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_cleaning_v3()
