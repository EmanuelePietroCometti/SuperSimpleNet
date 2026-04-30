import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve
)

def evaluate_and_save_metrics(y_true, y_probs, output_dir="./eval_results", threshold="auto"):
    """
    Evaluates model predictions, automatically finds the best threshold (if set to 'auto'),
    prints/saves a classification report, and generates/saves both a Confusion Matrix and an AUROC plot.
    
    Args:
        y_true (list or array): Ground truth binary labels (0 or 1).
        y_probs (list or array): Predicted probabilities for the positive class.
        output_dir (str): Path to the directory where results will be saved.
        threshold (float or str): Probability threshold or "auto" to maximize F1-Score.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================
    # Automatic Threshold Calculation
    # ==========================================
    if threshold == "auto":
        # Calculate precision, recall, and thresholds
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        
        # Calculate F1 score for each threshold (ignoring the last precision/recall value to match thresholds array size)
        # Added 1e-10 to avoid division by zero
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        
        # Find the index of the maximum F1 score
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        
        print(f"\n[*] Auto-threshold calculated (Max F1-Score): {optimal_threshold:.4f} (F1: {f1_scores[best_idx]:.4f})")
        threshold = optimal_threshold
    else:
        print(f"\n[*] Using manual threshold: {threshold}")

    # Convert probabilities to discrete binary predictions based on the final threshold
    y_pred = (np.array(y_probs) >= threshold).astype(int)
    
    # ==========================================
    # Classification Report (Screen & File)
    # ==========================================
    report_text = classification_report(y_true, y_pred)
    
    print("\n" + "="*35)
    print("       CLASSIFICATION REPORT       ")
    print("="*35)
    print(report_text)
    
    report_path = os.path.join(output_dir, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write(f"=== Model Evaluation Report ===\n")
        f.write(f"Applied Threshold: {threshold:.4f}\n\n")
        f.write(report_text)
    print(f"[+] Report saved to: {report_path}")

    # ==========================================
    # Confusion Matrix (Plot & Save)
    # ==========================================
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 14})
    plt.title(f"Confusion Matrix (Threshold: {threshold:.2f})", pad=15)
    plt.xlabel("Predicted Label", labelpad=10)
    plt.ylabel("True Label", labelpad=10)
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)  
    plt.close()  
    print(f"[+] Confusion matrix saved to: {cm_path}")

    # ==========================================
    # AUROC Curve (Plot & Save)
    # ==========================================
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC)', pad=15)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(output_dir, "auroc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[+] AUROC plot saved to: {roc_path}\n")