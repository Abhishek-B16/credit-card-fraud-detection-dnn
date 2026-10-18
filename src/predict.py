"""
Inference module for real-time transaction scoring using trained FraudDNN.
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
from src.models import FraudDNN

class FraudPredictor:
    def __init__(self, model_path="models/best_dnn_model.pth", preprocessor_path="models/preprocessors.joblib"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                f"Trained model artifacts not found. Please train the model first by running:\n"
                f"python -m src.train"
            )
            
        self.preprocessors = joblib.load(preprocessor_path)
        self.scaler_amount = self.preprocessors['scaler_amount']
        self.scaler_time = self.preprocessors['scaler_time']
        self.feature_names = self.preprocessors['feature_names']
        
        # Load weights
        self.model = FraudDNN(input_dim=len(self.feature_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, transaction_dict_or_df, threshold=0.5):
        """
        Takes raw transaction features (dict or DataFrame with Time, V1..V28, Amount)
        and returns prediction results.
        """
        if isinstance(transaction_dict_or_df, dict):
            df = pd.DataFrame([transaction_dict_or_df])
        else:
            df = transaction_dict_or_df.copy()
            
        # Ensure required columns exist
        missing = [f"V{i}" for i in range(1, 29) if f"V{i}" not in df.columns]
        if missing or 'Time' not in df.columns or 'Amount' not in df.columns:
            raise ValueError(f"Transaction data missing essential features: {missing}")

        # Preprocessing & scaling
        df['scaled_time'] = self.scaler_time.transform(df[['Time']])
        # Clip extreme outlier amount to prevent single-feature dominance over PCA vectors
        df['scaled_amount'] = np.clip(self.scaler_amount.transform(df[['Amount']]), -3.0, 5.0)
        
        X = df[self.feature_names].values
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            probs = self.model.predict_proba(tensor_x).cpu().numpy().flatten()
            
        preds = (probs >= threshold).astype(int)
        
        results = []
        for i, (prob, pred) in enumerate(zip(probs, preds)):
            risk_level = "CRITICAL / FRAUD" if pred == 1 else ("SUSPICIOUS" if prob > 0.25 else "NORMAL")
            results.append({
                "transaction_index": i,
                "fraud_probability": float(prob),
                "is_fraud": bool(pred == 1),
                "risk_level": risk_level,
                "confidence_score": float(prob if pred == 1 else (1 - prob))
            })
            
        return results if len(results) > 1 else results[0]


if __name__ == "__main__":
    predictor = FraudPredictor()
    # Test on a dummy transaction
    dummy_tx = {'Time': 4000.0, 'Amount': 149.99}
    for i in range(1, 29):
        dummy_tx[f'V{i}'] = 0.0
    res = predictor.predict(dummy_tx)
    print("Test inference result:", res)
