"""
Evaluation metrics, plotting utilities, and artifact helpers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score
)
import torch

def evaluate_predictions(y_true, y_probs, threshold=0.5):
    """Calculates comprehensive classification metrics for imbalanced fraud data."""
    y_pred = (y_probs >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    report = classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"], output_dict=True, zero_division=0)
    
    return {
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_score": f1,
        "classification_report": report,
        "threshold": threshold
    }


def plot_confusion_matrix(cm, save_path="reports/figures/confusion_matrix.png"):
    """Saves a styled confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["Predicted Legit", "Predicted Fraud"],
        yticklabels=["Actual Legit", "Actual Fraud"]
    )
    plt.title("Confusion Matrix - Fraud Detection DNN", fontsize=13, fontweight='bold', pad=12)
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Confusion matrix saved at: {save_path}")


def plot_roc_pr_curves(y_true, y_probs, save_path="reports/figures/roc_pr_curves.png"):
    """Plots ROC Curve and Precision-Recall Curve side-by-side."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. ROC Curve
    axes[0].plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'DNN (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=11)
    axes[0].set_title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)
    
    # 2. Precision-Recall Curve
    axes[1].plot(recall, precision, color='#d62728', lw=2.5, label=f'DNN (PR-AUC = {pr_auc:.4f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (Sensitivity)', fontsize=11)
    axes[1].set_ylabel('Precision', fontsize=11)
    axes[1].set_title('Precision-Recall Curve (Imbalanced Metric)', fontsize=12, fontweight='bold')
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Performance curves saved at: {save_path}")


def plot_training_history(train_losses, val_losses, save_path="reports/figures/training_history.png"):
    """Plots training and validation loss curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(train_losses, label="Train Loss", color="#2ca02c", lw=2)
    plt.plot(val_losses, label="Validation Loss", color="#ff7f0e", lw=2)
    plt.title("Deep Neural Network Loss Progression", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Training history saved at: {save_path}")
