"""
Feature Engineering utilities for CKD Prediction Pipeline.

Creates clinically meaningful derived features to improve class separability,
especially between Stable CKD and Death/Progression.
"""

import numpy as np


def engineer_features(df):
    """
    Add engineered features to a DataFrame.
    Works for both training data and single-patient prediction in app.py.
    
    Args:
        df: pandas DataFrame with the raw 14 features
    
    Returns:
        df with additional engineered columns
    """
    df = df.copy()

    # 1. BUN/Creatinine Ratio — classic pre-renal failure marker
    #    High ratio (>20) suggests pre-renal causes (dehydration, heart failure → death)
    df['bun_creatinine_ratio'] = df['bun_mgdl'] / (df['serum_creatinine_mgdl'] + 0.01)

    # 2. Calcium-Phosphate Product — cardiovascular death predictor in CKD
    #    Ca×P > 55 is associated with vascular calcification and mortality
    df['calcium_phosphate_product'] = df['calcium_mgdl'] * df['phosphate_mgdl']

    # 3. Renal Risk Score — composite deterioration index
    #    Amplifies the tiny individual differences across 4 correlated features
    df['renal_risk_score'] = (
        df['serum_creatinine_mgdl'] * df['potassium_meql'] * df['phosphate_mgdl']
    ) / (df['egfr'] + 0.01)

    # 4. Log-transformed UACR — reduces right-skew for better tree splits
    df['uacr_log'] = np.log1p(df['uacr_mg_g'])

    return df
