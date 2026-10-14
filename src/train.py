"""
End-to-End Training Script for Deep Neural Network Fraud Detection.
Saves model checkpoints, evaluation metrics, and diagnostic plots.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure utf-8 encoding for cross-platform terminals
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from src.data_loader import load_and_preprocess_data
from src.models import FraudDNN, FraudAutoencoder
from src.utils import evaluate_predictions, plot_confusion_matrix, plot_roc_pr_curves, plot_training_history

def train_dnn(epochs=15, batch_size=256, lr=1e-3, data_path="data/creditcard.csv", save_dir="models", epoch_callback=None):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"[INFO] Training Deep Neural Network on device: {device}")
    print(f"=======================================================")
    
    # 1. Load Data
    train_loader, val_loader, test_loader, (X_test, y_test), metadata = load_and_preprocess_data(
        data_path=data_path, batch_size=batch_size
    )
    
    # Save scalers for live inference
    joblib.dump({
        'scaler_amount': metadata['scaler_amount'],
        'scaler_time': metadata['scaler_time'],
        'feature_names': metadata['feature_names']
    }, os.path.join(save_dir, "preprocessors.joblib"))
    
    # 2. Initialize Model
    model = FraudDNN(input_dim=metadata['num_features']).to(device)
    
    # 3. Loss & Optimizer (Weighted BCE with pos_weight for class imbalance)
    pos_weight = torch.tensor([min(metadata['pos_weight'], 100.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, "best_dnn_model.pth")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * len(batch_x)
            
        avg_train_loss = total_train_loss / metadata['n_train']
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                total_val_loss += loss.item() * len(batch_x)
                
        avg_val_loss = total_val_loss / metadata['n_val']
        scheduler.step(avg_val_loss)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if epoch_callback is not None:
            epoch_callback(epoch, avg_train_loss, avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            
    print(f"\n[Model] Best DNN weights saved to: {best_model_path}")
    
    # 4. Final Evaluation on Holdout Test Set
    print("\n--- Evaluating Best Model on Test Set ---")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        probs = model.predict_proba(test_tensor).cpu().numpy().flatten()
        
    metrics = evaluate_predictions(y_test, probs, threshold=0.5)
    
    print(f"[RESULT] Test ROC-AUC Score : {metrics['roc_auc']:.4f}")
    print(f"[RESULT] Test PR-AUC Score  : {metrics['pr_auc']:.4f}")
    print(f"[RESULT] Test F1-Score      : {metrics['f1_score']:.4f}")
    print("\nClassification Report:\n")
    for cls_name, vals in metrics['classification_report'].items():
        if isinstance(vals, dict):
            print(f"  {cls_name:<12}: Precision={vals['precision']:.3f}, Recall={vals['recall']:.3f}, F1={vals['f1-score']:.3f}")
            
    # 5. Generate Diagnostic Plots
    plot_training_history(train_losses, val_losses)
    plot_confusion_matrix(metrics['confusion_matrix'])
    plot_roc_pr_curves(y_test, probs)
    
    metrics['train_losses'] = train_losses
    metrics['val_losses'] = val_losses
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fraud Detection DNN")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv", help="Path to creditcard.csv")
    args = parser.parse_args()
    
    train_dnn(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, data_path=args.data_path)
