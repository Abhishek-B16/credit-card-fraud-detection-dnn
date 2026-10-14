# 🎙️ Credit Card Fraud Detection using Deep Neural Network
## Complete Presentation Script & Viva Cheatsheet
**Author:** Abhishek  
**Project:** Credit Card Fraud Detection using Deep Neural Network (PyTorch)

---

## 📌 Quick Summary To Remember Before You Start
* **Framework:** PyTorch (Deep Multi-Layer Perceptron / MLP).
* **Layers & Neurons:** 4 Layers: $30 \rightarrow 128 \rightarrow 64 \rightarrow 32 \rightarrow 16 \rightarrow 1$.
* **The Main Challenge:** Severe Class Imbalance (99.83% legitimate, only 0.17% fraud).
* **The Solution:** Cost-Weighted Binary Cross-Entropy Loss (penalty multiplier $\approx 588\times$).
* **Test Performance:** **ROC-AUC: 0.9999** | **PR-AUC: 0.9683** | **Fraud Recall: 100%**.
* **Live App URL:** `http://localhost:8502`

---

## 🎬 Step-by-Step Presentation Script

---

### Step 1: Introduction (Opening 45 Seconds)

**[WHAT TO DO ON SCREEN]**  
Have the browser open at `http://localhost:8502` showing the main dashboard.

**[WHAT TO SAY]**  
> *"Good morning / afternoon, Sir.*  
> *Today I am presenting my project: **Credit Card Fraud Detection using Deep Neural Network**.*  
>
> *Every day, millions of card transactions happen worldwide. When someone swipes their card, the bank's AI must decide in less than 5 milliseconds: is this transaction genuine, or is it a stolen card?*  
>
> *I built an end-to-end Deep Learning system using **PyTorch** that detects fraudulent transactions in real time with over 96% Precision-Recall AUC and zero missed frauds on our test set."*

---

### Step 2: The Core Problem (In Simple Words)

**[WHAT TO SAY]**  
> *"Sir, before looking at the model, the most important thing to understand is the **Class Imbalance Problem**.*  
>
> *In real life, out of every 10,000 transactions, around 9,983 are legitimate, and only about 17 are fraudulent. That means 99.8% of the data belongs to normal customers.*  
>
> *If a simple machine learning model just guesses 'Legitimate' for every single transaction, it will get a **99.8% accuracy**, but it will catch **zero frauds**. All the criminals would get away.*  
>
> *That is why accuracy is a misleading metric for fraud detection. In this project, our primary goal was **Recall**—catching 100% of frauds without blocking innocent customers."*

---

### Step 3: The Data & Privacy (Why Features are Named V1 to V28)

**[WHAT TO SAY]**  
> *"Sir, if we look at the dataset:*  
> *Due to European banking privacy laws like GDPR and PCI-DSS, banks cannot share real customer names, card numbers, or locations.*  
>
> *So the bank applied a technique called **PCA (Principal Component Analysis)**. PCA transforms the confidential user information into **28 mathematical numerical values, named V1 to V28**.*  
>
> *Only two real columns remain untouched:*  
> 1. *`Time` (seconds elapsed).*  
> 2. *`Amount` (dollar amount of purchase).*  
>
> *For data preprocessing, I used **RobustScaler** on the Amount. Financial amounts have huge outliers—like a $2 coffee versus a $10,000 diamond purchase. RobustScaler uses the median and interquartile range so extreme numbers don't distort normal data."*

---

### Step 4: The Deep Neural Network Architecture

**[WHAT TO SAY]**  
> *"Sir, in `src/models.py`, I built a 4-layer Deep Multi-Layer Perceptron in PyTorch:*  
>
> 1. *The **Input Layer** takes 30 features (Time, V1 to V28, and Amount).*  
> 2. *The **Hidden Layers** taper down: **128 neurons $\rightarrow$ 64 neurons $\rightarrow$ 32 neurons $\rightarrow$ 16 neurons**.*  
> 3. *The **Output Layer** has 1 neuron with a **Sigmoid** activation that outputs a risk probability from 0% to 100%.*  
>
> *To make training stable and prevent overfitting, I added three key techniques:*  
> - *First, **Batch Normalization** after every layer to keep the learning smooth and stable.*  
> - *Second, **LeakyReLU** activations with a small 0.1 slope so neurons never freeze or die.*  
> - *Third, **Dropout of 30%** so the network doesn't memorize the small number of fraud examples.*  
>
> *And most importantly, during training, I used **Weighted Binary Cross Entropy Loss** with a positive weight of **588x**. This tells PyTorch: missing a single fraud costs 588 times more loss than a false alarm. This forces the neural network to catch all frauds."*

---

### Step 5: Live Demonstration (Screen Walkthrough)

---

#### 🟢 Action 1: Test a Legitimate Purchase (Tab 1: Simulator)
**[WHAT TO DO]**  
1. Make sure you are on **Tab 1: 💳 Transaction Simulator**.  
2. Click the top button: **`☕ Normal Coffee ($4.50)`**.  

**[WHAT TO SAY]**  
> *"Sir, let's test a normal transaction live.*  
> *Here is a coffee purchase for $4.50 at 8:30 AM from a trusted mobile phone.*  
> *Looking at the risk gauge on the right: the neural network calculates a fraud risk of **under 1%**.*  
> *The green banner appears: **TRANSACTION APPROVED (LEGITIMATE)**. The card is cleared in 2.8 milliseconds."*

---

#### 🔴 Action 2: Test a Fraudulent Attack (Tab 1: Simulator)
**[WHAT TO DO]**  
Click the top button: **`🚨 Crypto Cashout ($4,800.00)`**.

**[WHAT TO SAY]**  
> *"Now, let's simulate a cyber attack.*  
> *Here is an unauthorized cashout of $4,800 at 3:20 AM from a foreign VPN proxy, with impossible travel velocity.*  
> *Watch the gauge: the risk score instantly jumps above **70% to 100%**.*  
> *The red banner triggers: **TRANSACTION BLOCKED (FRAUD DETECTED)**.*  
> *The bank automatically suspends the card and sends an alert to the cardholder."*

---

#### 📡 Action 3: Show the Live Bank Monitoring Stream (Tab 2)
**[WHAT TO DO]**  
Click on **Tab 2: 📡 Live Bank Feed**.

**[WHAT TO SAY]**  
> *"Sir, Tab 2 shows our real-time Bank Operations Stream.*  
> *In the top cards, you can see total transactions processed, frauds intercepted, and **$23,100 of customer money saved**.*  
> *Below, the table displays incoming card swipes from users around the world, with cardholders, amounts, and the neural network's instant verdict at **2.8 milliseconds latency**."*

---

#### 📁 Action 4: Show the Batch CSV Auditor (Tab 3)
**[WHAT TO DO]**  
Click on **Tab 3: 📁 Batch CSV Auditor**.  
Point to the buttons: **`⚡ Audit Sample 50`** and **`📥 Download CSV`**.

**[WHAT TO SAY]**  
> *"On Tab 3, we can audit hundreds of transactions at once from a file.*  
> *If I click 'Audit Sample 50', the Deep Neural Network evaluates 50 transactions simultaneously in milliseconds, flags all fraudulent anomalies, and gives an audit report.*  
> *We also have a 'Download CSV' button to export and inspect the data."*

---

#### 📊 Action 5: Show Model Performance & Confusion Matrix (Tab 4)
**[WHAT TO DO]**  
Click on **Tab 4: 📊 Performance & Analytics**.

**[WHAT TO SAY]**  
> *"On Tab 4, here are our holdout test set results:*  
> - *Our **ROC-AUC is 0.9999**, showing near-perfect class separation.*  
> - *Our **PR-AUC is 0.9683**, which is the gold standard metric for imbalanced fraud data.*  
> - *Our **Fraud Recall is 100%**, meaning every fraudulent sample in the test set was successfully caught.*  
> - *Here you can see the Confusion Matrix and ROC curves generated directly by our evaluation script."*

---

#### ⚡ Action 6: Show the Live Trainer (Tab 5)
**[WHAT TO DO]**  
Click on **Tab 5: ⚡ Model Trainer**.

**[WHAT TO SAY]**  
> *"Finally, Sir, in Tab 5, this is not just a static pre-recorded model.*  
> *We can adjust the Training Epochs and Learning Rate using these sliders, and click 'Start Model Training' to watch PyTorch optimize the weights and update the loss curve live on screen."*

---

### Step 6: Conclusion (Ending 30 Seconds)

**[WHAT TO SAY]**  
> *"To conclude, Sir:*  
> *1. We built a 4-layer regularized Deep Neural Network in PyTorch.*  
> *2. We solved the 99.8% class imbalance using Cost-Weighted Loss.*  
> *3. We achieved 0.9683 PR-AUC and 100% Fraud Recall.*  
> *4. We deployed it with a real-time web dashboard and containerized it with Docker.*  
> *The entire project is pushed to my GitHub repository with complete modular source code.*  
>
> *Thank you, Sir. I am ready for any questions."*

---

## ❓ Top 6 Questions Your Professor Might Ask (And How to Answer)

### 1. "Why Deep Learning instead of Logistic Regression or Random Forest?"
> **Answer:** *"Sir, fraud patterns are non-linear and high-dimensional. Logistic Regression is linear and misses complex interactions between time, amount, and PCA components. While Random Forest is good, our PyTorch Deep Neural Network with Batch Normalization and Dropout provides faster batch matrix inference (< 3 milliseconds), scales better to massive transaction streams, and doesn't require storing large tree files in memory."*

### 2. "Why is Accuracy a bad metric for fraud detection?"
> **Answer:** *"Sir, because 99.83% of transactions are legitimate. A model that blindly predicts 'Legitimate' for every swipe gets 99.83% accuracy, but catches zero frauds. That is why we evaluate using **PR-AUC (0.9683)** and **Recall (100%)**, which evaluate catching the rare fraud class."*

### 3. "What does `pos_weight` do in the loss function?"
> **Answer:** *"Sir, `pos_weight` is a penalty multiplier for the positive (fraud) class. Because there are ~588 legitimate samples for every 1 fraud, we set `pos_weight` to 588. This forces the gradient during backpropagation to penalize missing a fraud 588 times more heavily than a false alarm."*

### 4. "Why did you use RobustScaler for Amount instead of StandardScaler?"
> **Answer:** *"Sir, financial amounts have huge outliers (e.g. $1 vs $15,000). StandardScaler uses the mean and standard deviation, which get heavily skewed by extreme purchases. RobustScaler uses the median and interquartile range (IQR), which are resistant to outliers and keep normal purchases from being squashed to zero."*

### 5. "What is LeakyReLU and why not standard ReLU?"
> **Answer:** *"Sir, standard ReLU turns all negative inputs into zero. If a neuron's weights output negative values, its gradient becomes zero permanently—this is known as the 'Dying ReLU' problem. LeakyReLU gives negative values a small 0.1 slope, keeping gradient flow active across all neurons throughout training."*

### 6. "How does Docker help in this project?"
> **Answer:** *"Sir, our Docker container packages Python, PyTorch, and our pre-trained model weights (`models/best_dnn_model.pth`). Anyone can run `docker compose up` on any computer, and the dashboard launches immediately on port 8501 without needing to install libraries or retrain the model."*
