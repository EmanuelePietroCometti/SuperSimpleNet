import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

def evaluate_and_save_metrics(
    y_true, 
    y_probs, 
    output_dir="./eval_results", 
    threshold="auto", 
    image_paths=None
):
    """
    Evaluates model predictions, automatically finds the best threshold,
    prints/saves a classification report, generates/saves Confusion Matrix and AUROC plot,
    and optionally categorizes and saves original images based on predictions.
    
    Args:
        y_true (list or array): Ground truth binary labels (0 or 1).
        y_probs (list or array): Predicted probabilities for the positive class.
        output_dir (str): Path to the directory where results will be saved.
        threshold (float or str): Probability threshold or "auto" to maximize F1-Score.
        image_paths (list, optional): List of image paths. If provided, images are sorted and saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # ==========================================
    # Automatic threshol calculation
    # ==========================================
    if threshold == "auto":
        # Calculate precision, recall, and thresholds
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        
        # Calculate F1 score for each threshold (ignoring the last value to match array sizes)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        
        # Find the index of the maximum F1 score
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        
        print(f"\n[*] Auto-threshold calculated (Max F1-Score): {optimal_threshold:.4f} (F1: {f1_scores[best_idx]:.4f})")
        threshold = optimal_threshold
    else:
        print(f"\n[*] Using manual threshold: {threshold}")

    # Convert probabilities to discrete binary predictions based on the final threshold
    y_pred = (y_probs >= threshold).astype(int)
    
    # ==========================================
    # Classification report (screen and file)
    # ==========================================
    report_text = classification_report(y_true, y_pred)
    
    print("\n" + "="*35)
    print("       CLASSIFICATION REPORT       ")
    print("="*35)
    print(report_text)
    
    report_path = output_dir / "metrics_report.txt"
    with open(report_path, "w") as f:
        f.write(f"=== Model Evaluation Report ===\n")
        f.write(f"Applied Threshold: {threshold:.4f}\n\n")
        f.write(report_text)
    print(f"[+] Report saved to: {report_path}")

    # ==========================================
    # Confusion matrix (plot and save)
    # ==========================================
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 14})
    plt.title(f"Confusion Matrix (Threshold: {threshold:.2f})", pad=15)
    plt.xlabel("Predicted Label", labelpad=10)
    plt.ylabel("True Label", labelpad=10)
    plt.tight_layout()
    
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)  
    plt.close()  
    print(f"[+] Confusion matrix saved to: {cm_path}")

    # ==========================================
    # AUROC curve (plot and save)
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
    
    roc_path = output_dir / "auroc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[+] AUROC plot saved to: {roc_path}\n")

    # ==========================================
    # Categorize and save images
    # ==========================================
    if image_paths is not None:
        print("[*] Sorting classification images based on predictions...")
        images_dir = output_dir / "images_by_prediction"
        
        # Define subdirectories 
        categories = {
            "TN": images_dir / "predicted_good" / "correct",   # Actual 0, Predicted 0
            "FN": images_dir / "predicted_good" / "wrong",     # Actual 1, Predicted 0
            "TP": images_dir / "predicted_reject" / "correct", # Actual 1, Predicted 1
            "FP": images_dir / "predicted_reject" / "wrong",   # Actual 0, Predicted 1
        }

        for path in categories.values():
            path.mkdir(parents=True, exist_ok=True)

        # Iterate using the already computed discrete predictions (y_pred)
        for img_path_str, label, pred in zip(image_paths, y_true, y_pred):
            img_path = Path(img_path_str)
            
            if label == 0 and pred == 0:
                dest_dir = categories["TN"]
            elif label == 1 and pred == 1:
                dest_dir = categories["TP"]
            elif label == 0 and pred == 1:
                dest_dir = categories["FP"]
            elif label == 1 and pred == 0:
                dest_dir = categories["FN"]
            
            # Prepend parent folder to avoid filename collisions
            dest_filename = f"{img_path.parent.name}_{img_path.name}"
            shutil.copy(img_path, dest_dir / dest_filename)
            
        print(f"[+] Images successfully categorized and saved in: {images_dir}\n")
        
    return threshold