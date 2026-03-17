import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_feature_importance(model, feature_names, results_dir: str = 'results', figures_dir: str = 'figures') -> pd.DataFrame:
    """
    Extracts, saves, and plots the top 10 most important biomarkers.

    Args:
        model: Trained RandomForestClassifier with feature_importances_.
        feature_names: List or Index of feature column names.
        results_dir: Directory to save CSV results.
        figures_dir: Directory to save the feature importance plot.

    Returns:
        DataFrame with all features ranked by importance.
    """
    # 1. Feature Importance
    importances = model.feature_importances_

    # Safety check: ensure feature names match model features
    if len(importances) != len(feature_names):
        raise ValueError("Number of feature names does not match number of model importances.")

    feat_imp_df = pd.DataFrame({
        'Biomarker': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # 2. Select Top 10
    top_10 = feat_imp_df.head(10).sort_values(by='Importance', ascending=True)
    
    # 3. Save CSV
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'feature_importance_rf.csv')
    feat_imp_df.to_csv(csv_path, index=False)
    top10_path = os.path.join(results_dir, 'top10_feature_importance_rf.csv')
    top_10.to_csv(top10_path, index=False)
    print(f"\n=== Step 8 & 9: Feature Importance ===")
    print(f"Full feature importance saved to {csv_path}")
    
    # 4. Horizontal Bar Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Biomarker', data=top_10, color='steelblue')
    
    plt.title('Top Biomarkers Predicting CKD Outcomes (Random Forest)', fontsize=14, pad=15)
    plt.xlabel('Feature Importance Score', fontsize=12)
    plt.ylabel('Biomarker / Predictor Variable', fontsize=12)
    
    # Save Figure
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, 'random_forest_feature_importance.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    print(f"Top 10 feature importance plot saved to {fig_path}")
    print(f"Top 10 feature importance table saved to {top10_path}")
    
    return feat_imp_df
