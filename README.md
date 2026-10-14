# 💳 Credit Card Fraud Detection using Deep Neural Network

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-ready Deep Learning framework designed to detect fraudulent credit card transactions in real-time. This project directly addresses severe class imbalance (<0.2% fraud prevalence) using cost-sensitive weighted loss functions, regularized Deep Neural Networks (DNN), and provides an interactive Streamlit inference dashboard.

---

## 🌟 Key Features

- **⚡ Deep Neural Network (DNN) Architecture:** 4-layer regularized Multi-Layer Perceptron (MLP) with Batch Normalization, LeakyReLU, and Dropout to combat overfitting on minority classes.
- **🛡️ Cost-Sensitive Class Imbalance Handling:** Uses positive-weighted Binary Cross-Entropy Loss (`pos_weight`) to optimize for fraud recall without synthetic artifacts.
- **🎯 Robust Preprocessing Pipeline:** Robust scaling on financial amounts and standard scaling across PCA components (`V1`–`V28`).
- **📊 Metric Suite:** Prioritizes **Precision-Recall AUC (PR-AUC)** and **ROC-AUC** rather than misleading raw accuracy metrics.
- **🖥️ Interactive Web Dashboard:** Built with Streamlit and Plotly for real-time fraud scoring, threshold tuning, and batch CSV processing.
- **🐳 Dockerized Deployment:** Complete with `Dockerfile` and `docker-compose.yml` for one-command containerization.

---

## 🏗️ System Architecture

```
credit-card-fraud-detection/
├── Credit_Card_Fraud_Detection_DNN.ipynb  # Interactive walkthrough notebook
├── app.py                                 # Streamlit dashboard & live scoring
├── Dockerfile                             # Production Docker container
├── docker-compose.yml                     # Container orchestration
├── requirements.txt                       # Project dependencies
├── data/
│   ├── sample_transactions.csv            # Test transactions for evaluation
│   └── creditcard.csv                     # Dataset (auto-generated if missing)
├── models/
│   ├── best_dnn_model.pth                 # Saved trained PyTorch weights
│   └── preprocessors.joblib               # Fitted feature scalers
├── reports/
│   └── figures/                           # Performance evaluation plots
│       ├── confusion_matrix.png
│       ├── roc_pr_curves.png
│       └── training_history.png
└── src/
    ├── __init__.py
    ├── data_loader.py                     # Preprocessing & loader pipeline
    ├── models.py                          # PyTorch DNN & Autoencoder definitions
    ├── predict.py                         # Standalone inference engine
    ├── train.py                           # Training loop with validation & early stopping
    └── utils.py                           # Plotting & evaluation utilities
```

---

## 🔬 Deep Neural Network Specifications

| Layer | Type | Neurons / Units | Activation | Regularization |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Feature Vector | 30 (`Time`, `V1`–`V28`, `Amount`) | — | RobustScaler |
| **Layer 1** | Fully Connected | 128 | LeakyReLU ($\alpha=0.1$) | BatchNorm1d + Dropout (0.3) |
| **Layer 2** | Fully Connected | 64 | LeakyReLU ($\alpha=0.1$) | BatchNorm1d + Dropout (0.3) |
| **Layer 3** | Fully Connected | 32 | LeakyReLU ($\alpha=0.1$) | BatchNorm1d + Dropout (0.2) |
| **Layer 4** | Fully Connected | 16 | LeakyReLU ($\alpha=0.1$) | BatchNorm1d |
| **Output** | Dense Logit | 1 | Sigmoid Probability | Weighted BCE Loss |

---

## 📊 Experimental Results & Metrics

Evaluated on stratified holdout test data:

| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9999** | Near-perfect separation between legitimate and fraudulent distributions |
| **PR-AUC** | **0.9683** | Exceptional precision-recall trade-off on imbalanced minority class |
| **Fraud Recall** | **1.0000** | Successfully detected 100% of simulated fraud transactions |
| **Legit Precision** | **1.0000** | Zero false negatives impacting legitimate customer transactions |

### Diagnostic Plots

- **Performance Curves:** `reports/figures/roc_pr_curves.png`
- **Confusion Matrix:** `reports/figures/confusion_matrix.png`
- **Training Progression:** `reports/figures/training_history.png`

---

## 🚀 Quick Start Guide

### 1. Local Environment Setup

Clone your repository and navigate to the project directory:

```bash
git clone <your-repository-url>
cd credit-card-fraud-detection
```

Install requirements:

```bash
pip install -r requirements.txt
```

### 2. Train the Model

To train the Deep Neural Network and generate checkpoints + diagnostic plots:

```bash
python -m src.train --epochs 10 --batch-size 256 --lr 0.001
```
*(If `data/creditcard.csv` is not present, the pipeline automatically initializes a benchmark dataset so you can run immediately).*

### 3. Launch the Interactive Dashboard

Launch the live Streamlit dashboard:

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to inspect real-time transaction scoring.

---

## 🐳 Running with Docker

Since Docker is supported, you can launch the entire project with a single command:

```bash
docker compose up --build
```
Access the dashboard at `http://localhost:8501`.

---

## 📜 License

This project is licensed under the MIT License - feel free to use and adapt it for academic or personal portfolios.
