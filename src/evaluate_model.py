from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_and_plot(model, X_test, y_test, figures_dir='figures'):
    """
    Evaluates the model and saves the confusion matrix heatmap.
    """
    # 预测 (Prediction)
    y_pred = model.predict(X_test)
    
    print("\n=== Step 6 & 7: Model Evaluation & Visualization ===")
    
    # Metrics
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=[0,1,2], target_names=["Stable CKD", "Death", "ESRD"]))
    
    # 1. Create Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])
    class_labels = ["Stable CKD", "Death", "ESRD"]
    
    # 2. Plotting (Publication Quality)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title('Confusion Matrix for CKD Outcome Prediction', fontsize=16, pad=20)
    plt.xlabel('Predicted Outcome', fontsize=12)
    plt.ylabel('True Outcome', fontsize=12)
    
    # Save Figure
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, 'confusion_matrix_rf.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    print(f"Confusion matrix saved to {fig_path}")
