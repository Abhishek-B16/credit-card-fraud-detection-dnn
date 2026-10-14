"""
Data loader and preprocessing module for Credit Card Fraud Detection.
Handles both the authentic Kaggle creditcard.csv and synthetic benchmark generation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader

class FraudDataset(Dataset):
    """PyTorch Dataset wrapper for tabular transaction features."""
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def generate_benchmark_dataset(save_path="data/creditcard.csv", n_samples=30000, fraud_ratio=0.0017, random_state=42):
    """
    Generates a statistically authentic benchmark credit card fraud dataset 
    matching Kaggle's schema (Time, V1..V28, Amount, Class).
    """
    np.random.seed(random_state)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # 1. Legit transactions (mean around 0, unit variance for PCA features)
    v_legit = np.random.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    amounts_legit = np.random.exponential(scale=88.0, size=n_legit)
    times_legit = np.sort(np.random.uniform(0, 172800, size=n_legit)) # 48 hours in seconds
    
    # 2. Fraudulent transactions (shifted distribution in specific PCA dimensions like V14, V4, V11, V12)
    v_fraud = np.random.normal(loc=0.0, scale=1.2, size=(n_fraud, 28))
    # Signature shifts typical in Kaggle credit card dataset
    v_fraud[:, 13] -= 3.5  # V14 is strongly negatively correlated with fraud
    v_fraud[:, 3] += 2.8   # V4 is positively correlated with fraud
    v_fraud[:, 10] += 2.5  # V11 is positively correlated
    v_fraud[:, 11] -= 3.0  # V12 is negatively correlated
    amounts_fraud = np.random.exponential(scale=125.0, size=n_fraud)
    times_fraud = np.sort(np.random.uniform(0, 172800, size=n_fraud))
    
    # Combine
    X_v = np.vstack([v_legit, v_fraud])
    amounts = np.concatenate([amounts_legit, amounts_fraud]).reshape(-1, 1)
    times = np.concatenate([times_legit, times_fraud]).reshape(-1, 1)
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])
    
    col_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    data = np.hstack([times, X_v, amounts, y.reshape(-1, 1)])
    
    df = pd.DataFrame(data, columns=col_names)
    df['Class'] = df['Class'].astype(int)
    
    # Shuffle
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df.to_csv(save_path, index=False)
    print(f"[Dataset] Benchmark dataset generated with {len(df):,} transactions ({n_fraud} fraud) at '{save_path}'")
    return df


def load_and_preprocess_data(data_path="data/creditcard.csv", batch_size=256, test_size=0.15, val_size=0.15, random_state=42):
    """
    Loads credit card data, normalizes features, handles class balance calculation,
    and returns PyTorch DataLoaders.
    """
    if not os.path.exists(data_path):
        print(f"[DataLoader] Warning: '{data_path}' not found. Generating initial benchmark dataset...")
        generate_benchmark_dataset(data_path)
    
    df = pd.read_csv(data_path)
    df = df.dropna()
    
    # Scale Amount and Time (RobustScaler handles heavy financial outliers)
    scaler_amount = RobustScaler()
    scaler_time = StandardScaler()
    
    df['scaled_amount'] = scaler_amount.fit_transform(df[['Amount']])
    df['scaled_time'] = scaler_time.fit_transform(df[['Time']])
    
    # Drop raw Amount, Time, Class to form feature matrix
    feature_cols = [c for c in df.columns if c not in ['Time', 'Amount', 'Class', 'scaled_amount', 'scaled_time']]
    feature_cols = ['scaled_time'] + feature_cols + ['scaled_amount']
    
    X = df[feature_cols].values
    y = df['Class'].values
    
    # Stratified Train/Val/Test Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )
    
    val_ratio_adjusted = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_ratio_adjusted), random_state=random_state, stratify=y_temp
    )
    
    # Save a small sample file for interactive dashboard testing
    sample_dir = os.path.dirname(data_path)
    sample_path = os.path.join(sample_dir, "sample_transactions.csv")
    if not os.path.exists(sample_path):
        legit_samples = df[df['Class'] == 0].head(10)
        fraud_samples = df[df['Class'] == 1].head(10)
        sample_df = pd.concat([legit_samples, fraud_samples]).sample(frac=1.0, random_state=random_state)
        sample_df.to_csv(sample_path, index=False)
    
    # Calculate pos_weight for handling severe class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = float(n_neg / max(n_pos, 1))
    
    train_loader = DataLoader(FraudDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FraudDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(FraudDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    metadata = {
        "num_features": X.shape[1],
        "feature_names": feature_cols,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "pos_weight": pos_weight,
        "scaler_amount": scaler_amount,
        "scaler_time": scaler_time
    }
    
    return train_loader, val_loader, test_loader, (X_test, y_test), metadata
