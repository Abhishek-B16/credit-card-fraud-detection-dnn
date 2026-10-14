# 📊 10-Slide Presentation: Credit Card Fraud Detection using Deep Neural Network
**Presenter:** Abhishek  
**Framework:** PyTorch • Python 3.13 • Streamlit • Docker  
**Repository:** https://github.com/Abhishek-B16/credit-card-fraud-detection-dnn  

---

## Slide 1: Title & Overview
- **Title:** Credit Card Fraud Detection using Deep Neural Network
- **Subtitle:** Real-Time Cost-Sensitive Deep Learning for Financial Anomaly Detection
- **Presenter:** Abhishek
- **Core Technology:** PyTorch, Python 3.13, Streamlit, Docker
- **Objective:** Millisecond fraud scoring (<3ms) with 100% fraud recall under extreme class imbalance.

---

## Slide 2: Deep Learning 101 — Fundamentals from Scratch
- **Artificial Neuron:** Inputs ($x_i$) $\times$ Weights ($w_i$) + Bias ($b$) $\rightarrow$ Activation $f(z)$.
- **Forward Pass:** Input data flows through hidden layers to compute an output probability.
- **Loss Function:** Quantifies the prediction error against the true label.
- **Backpropagation:** Uses the Chain Rule of calculus to compute loss gradients with respect to each weight.
- **Optimization:** Gradient descent updates weights ($w \leftarrow w - \eta \nabla \mathcal{L}$) to minimize loss.

---

## Slide 3: The Problem Statement — The 99.8% Imbalance Trap
- **Digital Payment Scale:** Millions of global swipes per minute; decisions required in $<5\text{ ms}$.
- **Extreme Class Imbalance:**
  - Legitimate: **284,315** transactions (99.83%)
  - Fraudulent: **492** transactions (0.17%)
  - Ratio: $\approx \mathbf{588 : 1}$
- **The Accuracy Fallacy:** A model guessing "Legitimate" for 100% of data gets 99.83% accuracy but catches zero frauds.
- **Goal:** Maximize **Fraud Recall** and **Precision-Recall AUC (PR-AUC)** rather than raw accuracy.

---

## Slide 4: Dataset & Anonymization Architecture
- **Data Source:** Benchmark European cardholder dataset (284,807 transactions over 48 hours).
- **Privacy Compliance (GDPR & PCI-DSS):** Card numbers, CVVs, and names cannot be disclosed.
- **PCA Transformation:** Transformed confidential attributes into 28 orthogonal numerical features ($V_1$ to $V_{28}$).
- **Un-transformed Columns:** `Time` (seconds elapsed), `Amount` (dollar purchase value), `Class` ($0$ or $1$).
- **Partitioning:** Stratified 70% Train, 15% Validation, 15% Test split.

---

## Slide 5: Data Preprocessing & Scaling Strategy
- **Why Standard Scaling Fails:** Financial amounts range from $0.50 to $25,000+. Mean and standard deviation get distorted by extreme outliers.
- **Our Solution (`RobustScaler` on Amount):** Scales using the Median and Interquartile Range (IQR = $Q_3 - Q_1$). Immune to extreme outliers.
- **`StandardScaler` on Time:** Centers 48-hour time progression to zero-mean and unit variance.
- **Persistence:** Serialized via `Joblib` (`models/preprocessors.joblib`) for zero-latency live inference.

---

## Slide 6: Proposed Deep Neural Network (DNN) Architecture
- **Model Type:** 4-Layer Feedforward Multi-Layer Perceptron (PyTorch `FraudDNN`).
- **Bottleneck Funnel Design:**
  - Input Layer: 30 features
  - Layer 1: 128 neurons (complex non-linear feature interaction learning)
  - Layer 2: 64 neurons (latent representation refinement)
  - Layer 3: 32 neurons (noise compression)
  - Layer 4: 16 neurons (decision bottleneck)
  - Output Layer: 1 neuron with Sigmoid activation ($\sigma(z) \in [0.0, 1.0]$).

---

## Slide 7: Advanced Deep Learning Techniques Used
- **Batch Normalization (`BatchNorm1d`):** Stabilizes internal covariate shift across mini-batches.
- **`LeakyReLU(alpha=0.1)`:** Eliminates the "Dying ReLU" problem by allowing small negative gradients.
- **`Dropout(0.30)`:** Deactivates 30% of connections per step to prevent overfitting on the few fraud samples.
- **Optimizer & Scheduler:** AdamW with weight decay ($10^{-4}$) and `ReduceLROnPlateau` learning rate scheduler.

---

## Slide 8: Cost-Sensitive Loss Function (Weighted BCE)
- **Standard Loss:** Treats mistakes on legitimate and fraudulent transactions equally.
- **Our Weighted Loss:**
  $$\mathcal{L} = - [\mathbf{w}_{\text{pos}} \cdot y \log(p) + (1-y)\log(1-p)] \quad \text{where } \mathbf{w}_{\text{pos}} \approx \mathbf{588\times}$$
- **Impact:** Missing a fraud costs 588 times more loss than a false alarm. Forces the model to catch all frauds.

---

## Slide 9: Experimental Results & Academic Evaluation
- **Holdout Test Set Results:**
  - **ROC-AUC:** `0.9999` (near-perfect class discrimination)
  - **PR-AUC (Imbalanced):** `0.9683` (exceptional precision-recall trade-off)
  - **Fraud Recall:** `100.0%` (zero missed frauds on test partition)
  - **Inference Speed:** `2.8 ms` (production-ready)
- **Visual Diagnostics Generated:** Confusion Matrix heatmap, ROC/PR curves, and training history loss progression.

---

## Slide 10: System Deployment & Live Operations Dashboard
- **Pure Black Minimalist UI (`#000000`):** Seamless Streamlit dashboard running on `http://localhost:8502`.
- **5 Operational Modules:**
  1. *Real-World Transaction Simulator* (presets, amount, hour, merchant, impossible travel velocity).
  2. *Live Bank Operations Stream* (real-time card swipes, $23k+ money saved counter).
  3. *Batch CSV Auditor* (audit 50 to 1,000 transactions with sample download).
  4. *Performance & Analytics Center* (confusion matrix, ROC, PR-AUC).
  5. *Live PyTorch Model Trainer* (retrain weights and visualize live loss curves).
- **Containerization:** Packaged with `Dockerfile` and `docker-compose.yml` for instant 1-command startup.
