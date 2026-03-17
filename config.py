"""
Central configuration for all file paths used across the CKD pipeline.
Update paths here to change them everywhere.
"""

# Raw data
RAW_DATA_PATH = 'Data/Raw/ckd_community_screening.csv'

# Cleaned data
CLEANED_DATA_PATH = 'Data/Cleaned/ckd_cleaned_dataset_v3.csv'

# Train/test split directory
SPLIT_DIR = 'Data/Split'

# Graph output directory (used by data_cleaning.py)
GRAPH_DIR = 'Graph'

# Trained model
MODEL_PATH = 'models/random_forest_best_model.pkl'

# Output directories
MODELS_DIR = 'models'
FIGURES_DIR = 'figures'
RESULTS_DIR = 'results'
