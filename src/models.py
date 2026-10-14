"""
Deep Neural Network Architectures for Credit Card Fraud Detection:
1. Supervised Deep Neural Network (FraudDNN) with Batch Normalization & Dropout
2. Unsupervised Deep Autoencoder for Reconstruction Anomaly Detection
"""

import torch
import torch.nn as nn

class FraudDNN(nn.Module):
    """
    Supervised Deep Neural Network Classifier for Transaction Fraud Detection.
    Incorporates Batch Normalization and Dropout for regularization.
    """
    def __init__(self, input_dim=30, dropout_rate=0.3):
        super(FraudDNN, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            # Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            # Layer 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate * 0.7),
            
            # Layer 4 (Latent representation)
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            
            # Output Logit
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        """Returns fraud probability in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


class FraudAutoencoder(nn.Module):
    """
    Deep Autoencoder for Unsupervised Anomaly Detection.
    Trained strictly on legitimate transactions to minimize reconstruction error.
    Fraudulent transactions result in significantly higher reconstruction error.
    """
    def __init__(self, input_dim=30):
        super(FraudAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.Tanh(),
            nn.Linear(20, 10),
            nn.LeakyReLU(0.1),
            nn.Linear(10, 5), # Bottleneck latent dimension
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(5, 10),
            nn.LeakyReLU(0.1),
            nn.Linear(10, 20),
            nn.Tanh(),
            nn.Linear(20, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def compute_reconstruction_error(self, x):
        """Calculates Mean Squared Error per sample."""
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
            return mse
