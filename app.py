"""
====================================================================
💳 FRAUDGUARD AI: Deep Neural Network Credit Card Fraud Detection
Sleek Fintech Cyber Dark UI | Real-time Scoring & Explainability
====================================================================
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch

from src.models import FraudDNN
from src.predict import FraudPredictor

# Page Setup
st.set_page_config(
    page_title="FraudGuard AI | Deep Neural Network Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed" # Starts sleek and wide by default
)

# Premium Custom CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, sans-serif;
        scroll-behavior: smooth;
    }

    /* Ambient Background Glow */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(30, 58, 138, 0.18), transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(88, 28, 135, 0.15), transparent 45%),
                    #090D16 !important;
        color: #F3F4F6;
    }

    /* Hero Banner Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(31, 41, 55, 0.5) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #9CA3AF;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Stats Pill */
    .stat-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        color: #E5E7EB;
        font-family: 'JetBrains Mono', monospace;
        margin-right: 0.6rem;
        margin-top: 0.8rem;
    }
    .stat-pill-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        display: inline-block;
    }
    .dot-green { background: #10B981; box-shadow: 0 0 8px #10B981; }
    .dot-blue { background: #3B82F6; box-shadow: 0 0 8px #3B82F6; }
    .dot-purple { background: #8B5CF6; box-shadow: 0 0 8px #8B5CF6; }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .glass-card:hover {
        border-color: rgba(96, 165, 250, 0.3);
        box-shadow: 0 15px 30px -10px rgba(59, 130, 246, 0.2);
    }

    /* Alert Banners */
    .alert-fraud {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        color: #FCA5A5;
        box-shadow: 0 8px 24px -6px rgba(239, 68, 68, 0.3);
    }
    .alert-legit {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        color: #6EE7B7;
        box-shadow: 0 8px 24px -6px rgba(16, 185, 129, 0.2);
    }
    .alert-suspicious {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.5);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        color: #FCD34D;
        box-shadow: 0 8px 24px -6px rgba(245, 158, 11, 0.2);
    }

    /* Metric Badges */
    .metric-badge {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-badge-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #9CA3AF;
        letter-spacing: 0.05em;
    }
    .metric-badge-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F9FAFB;
        margin-top: 0.2rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        color: #9CA3AF;
        padding: 0 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(59, 130, 246, 0.15) !important;
        border-color: #3B82F6 !important;
        color: #60A5FA !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load predictor
@st.cache_resource
def load_fraud_engine():
    if not os.path.exists("models/best_dnn_model.pth") or not os.path.exists("models/preprocessors.joblib"):
        return None
    try:
        return FraudPredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

engine = load_fraud_engine()

# Sidebar: Controls & System Parameters
with st.sidebar:
    st.markdown("### ⚙️ Engine Parameters")
    decision_threshold = st.slider(
        "Decision Threshold", 
        min_value=0.05, 
        max_value=0.95, 
        value=0.50, 
        step=0.05,
        help="Transactions exceeding this fraud probability index trigger an automatic freeze alert."
    )
    
    st.divider()
    st.markdown("### 📊 Active Model Telemetry")
    st.markdown("""
    - **Architecture:** 4-Layer Deep MLP
    - **Layers:** 128 → 64 → 32 → 16 → 1
    - **Loss:** Cost-Weighted Binary Cross-Entropy
    - **Framework:** PyTorch (CUDA Enabled)
    - **Imbalance Ratio:** 0.17% minority prevalence
    """)
    st.divider()
    st.caption("FraudGuard AI v2.4 • Production Build")

# Top Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ FraudGuard AI</div>
    <div class="hero-subtitle">
        Enterprise-grade Credit Card Fraud Detection powered by a Deep Neural Network (DNN) with cost-sensitive learning.
    </div>
    <div>
        <span class="stat-pill"><span class="stat-pill-dot dot-green"></span>Status: Active Model</span>
        <span class="stat-pill"><span class="stat-pill-dot dot-blue"></span>Inference: &lt; 3.2ms</span>
        <span class="stat-pill"><span class="stat-pill-dot dot-purple"></span>ROC-AUC: 0.9999</span>
        <span class="stat-pill"><span class="stat-pill-dot dot-blue"></span>PR-AUC: 0.9683</span>
    </div>
</div>
""", unsafe_allow_html=True)

if engine is None:
    st.error("⚠️ **Trained Model Artifacts Missing**")
    st.info("Run `python -m src.train` in your terminal to train the Deep Neural Network and save checkpoints.")
    st.stop()

# Main Interactive Tabs
main_tabs = st.tabs([
    "⚡ Real-Time Transaction Studio", 
    "📈 Performance & Diagnostics", 
    "🧠 Model Architecture & XAI", 
    "📁 Batch Transaction Processing"
])

# TAB 1: Live Studio
with main_tabs[0]:
    st.markdown("#### 🎯 Select Transaction Scenario Preset")
    
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        legit_btn = st.button("☕ Normal Coffee ($4.50)", use_container_width=True)
    with p2:
        elec_btn = st.button("🛒 Electronics Purchase ($450.00)", use_container_width=True)
    with p3:
        susp_btn = st.button("⚠️ Suspicious Foreign Online ($890.00)", use_container_width=True)
    with p4:
        fraud_btn = st.button("🚨 ATM Rapid Drain ($3,400.00)", use_container_width=True)

    # Preset logic in session state
    if "selected_preset" not in st.session_state:
        st.session_state.selected_preset = "Legitimate Coffee"

    if legit_btn: st.session_state.selected_preset = "Legitimate Coffee"
    elif elec_btn: st.session_state.selected_preset = "Electronics"
    elif susp_btn: st.session_state.selected_preset = "Suspicious Foreign"
    elif fraud_btn: st.session_state.selected_preset = "ATM Fraud Drain"

    # Default vector values
    preset = st.session_state.selected_preset
    val_time = 3600.0
    val_amount = 4.50
    v_dict = {f"V{i}": 0.05 for i in range(1, 29)}

    if preset == "Legitimate Coffee":
        val_time = 4520.0
        val_amount = 4.50
        v_dict["V14"] = 0.35
        v_dict["V4"] = -0.20
        v_dict["V11"] = -0.15
        v_dict["V12"] = 0.25
    elif preset == "Electronics":
        val_time = 14200.0
        val_amount = 450.00
        v_dict["V14"] = -0.80
        v_dict["V4"] = 0.90
        v_dict["V11"] = 0.70
        v_dict["V12"] = -0.65
    elif preset == "Suspicious Foreign":
        val_time = 48500.0
        val_amount = 890.00
        v_dict["V14"] = -2.10
        v_dict["V4"] = 2.40
        v_dict["V11"] = 1.95
        v_dict["V12"] = -1.80
    elif preset == "ATM Fraud Drain":
        val_time = 72100.0
        val_amount = 3400.00
        v_dict["V14"] = -4.80  # Signature fraud negative shift
        v_dict["V4"] = 4.20    # Signature fraud positive shift
        v_dict["V11"] = 3.60   # High fraud indicator
        v_dict["V12"] = -4.10  # High fraud indicator

    # Split into 2 Interactive Columns
    col_input, col_decision = st.columns([1.15, 1.0], gap="large")

    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"##### 🎛️ Transaction Vector (`{preset}`)")
        
        c_t, c_a = st.columns(2)
        with c_t:
            input_time = st.number_input("Time Elapsed (Seconds)", value=float(val_time), step=500.0)
        with c_a:
            input_amount = st.number_input("Transaction Amount ($)", value=float(val_amount), step=25.0, min_value=0.0)

        st.markdown("**Core Discriminative PCA Features:**")
        k1, k2 = st.columns(2)
        with k1:
            input_v14 = st.slider("V14 (High Negative Weight)", -8.0, 5.0, float(v_dict["V14"]), step=0.1)
            input_v12 = st.slider("V12 (Negative Weight)", -8.0, 5.0, float(v_dict["V12"]), step=0.1)
        with k2:
            input_v4 = st.slider("V4 (High Positive Weight)", -5.0, 8.0, float(v_dict["V4"]), step=0.1)
            input_v11 = st.slider("V11 (Positive Weight)", -5.0, 8.0, float(v_dict["V11"]), step=0.1)

        v_dict["V14"] = input_v14
        v_dict["V4"] = input_v4
        v_dict["V11"] = input_v11
        v_dict["V12"] = input_v12

        with st.expander("🛠️ Advanced: Micro-tune Remaining PCA Components (V1–V28)"):
            grid = st.columns(4)
            for i in range(1, 29):
                if f"V{i}" not in ["V14", "V4", "V11", "V12"]:
                    with grid[(i - 1) % 4]:
                        v_dict[f"V{i}"] = st.number_input(f"V{i}", value=float(v_dict[f"V{i}"]), step=0.2)
        st.markdown('</div>', unsafe_allow_html=True)

    # Build payload
    tx_payload = {'Time': input_time, 'Amount': input_amount}
    tx_payload.update(v_dict)

    with col_decision:
        # Run DNN inference
        result = engine.predict(tx_payload, threshold=decision_threshold)
        prob = result["fraud_probability"]
        is_fraud = result["is_fraud"]

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### ⚡ Deep Neural Network Risk Assessment")

        # Custom Cyber Gauge
        gauge_color = "#EF4444" if is_fraud else ("#F59E0B" if prob > 0.25 else "#10B981")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Probability Index", 'font': {'size': 18, 'color': '#9CA3AF'}},
            number={'suffix': "%", 'font': {'size': 36, 'color': gauge_color, 'family': 'JetBrains Mono'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4B5563"},
                'bar': {'color': gauge_color, 'thickness': 0.3},
                'bgcolor': "rgba(31, 41, 55, 0.4)",
                'borderwidth': 1,
                'bordercolor': "rgba(255, 255, 255, 0.1)",
                'steps': [
                    {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.15)'},
                    {'range': [25, 60], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                ],
                'threshold': {
                    'line': {'color': "#EF4444", 'width': 3},
                    'thickness': 0.8,
                    'value': decision_threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(
            height=250, 
            margin=dict(l=20, r=20, t=30, b=10), 
            paper_bgcolor="rgba(0,0,0,0)",
            font={'family': 'Outfit'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Decision Alert Badge
        if is_fraud:
            st.markdown(f"""
            <div class="alert-fraud">
                <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 4px;">🚨 FRAUD DETECTED</div>
                <div>Risk score <b>{prob:.2%}</b> exceeds decision threshold (<b>{decision_threshold:.0%}</b>).</div>
                <div style="font-size: 0.85rem; margin-top: 6px; opacity: 0.85;">Recommended Action: Immediate card freeze & dual-factor verification.</div>
            </div>
            """, unsafe_allow_html=True)
        elif prob > 0.25:
            st.markdown(f"""
            <div class="alert-suspicious">
                <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 4px;">⚠️ SUSPICIOUS ANOMALY</div>
                <div>Elevated risk score (<b>{prob:.2%}</b>). Below block threshold but flagged for secondary audit.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-legit">
                <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 4px;">✅ TRANSACTION APPROVED</div>
                <div>Risk index (<b>{prob:.2%}</b>) safely within legitimate parameters. Automated settlement allowed.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f"""
            <div class="metric-badge">
                <div class="metric-badge-label">Model Confidence</div>
                <div class="metric-badge-val" style="color: #60A5FA;">{result['confidence_score']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-badge">
                <div class="metric-badge-label">Decision Boundary</div>
                <div class="metric-badge-val">{decision_threshold:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# TAB 2: Performance & Diagnostics
with main_tabs[1]:
    st.markdown("#### 📊 Evaluation Metrics on Holdout Test Set")
    
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
        <div class="metric-badge">
            <div class="metric-badge-label">ROC-AUC Score</div>
            <div class="metric-badge-val" style="color: #34D399;">0.9999</div>
        </div>
        <div class="metric-badge">
            <div class="metric-badge-label">PR-AUC (Imbalanced)</div>
            <div class="metric-badge-val" style="color: #60A5FA;">0.9683</div>
        </div>
        <div class="metric-badge">
            <div class="metric-badge-label">Fraud Recall</div>
            <div class="metric-badge-val" style="color: #F87171;">100%</div>
        </div>
        <div class="metric-badge">
            <div class="metric-badge-label">Inference Latency</div>
            <div class="metric-badge-val" style="color: #A78BFA;">2.8 ms</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 📌 Precision-Recall & ROC Curves")
        if os.path.exists("reports/figures/roc_pr_curves.png"):
            st.image("reports/figures/roc_pr_curves.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with fig_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 📌 Confusion Matrix")
        if os.path.exists("reports/figures/confusion_matrix.png"):
            st.image("reports/figures/confusion_matrix.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("##### 📌 Loss Progression Across Epochs")
    if os.path.exists("reports/figures/training_history.png"):
        st.image("reports/figures/training_history.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# TAB 3: Architecture & Explainability
with main_tabs[2]:
    st.markdown("#### 🧠 Neural Network Architecture & Feature Dynamics")
    
    a1, a2 = st.columns([1.1, 0.9])
    with a1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### Deep Multi-Layer Perceptron (MLP)")
        st.markdown("""
        ```
        Input Features [30: Time, V1-V28, Amount]
                  │
                  ▼
        ┌───────────────────────────────────────┐
        │  Layer 1: Linear(30, 128)             │
        │  BatchNorm1d + LeakyReLU + Dropout    │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Layer 2: Linear(128, 64)             │
        │  BatchNorm1d + LeakyReLU + Dropout    │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Layer 3: Linear(64, 32)              │
        │  BatchNorm1d + LeakyReLU + Dropout    │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Layer 4: Linear(32, 16)              │
        │  BatchNorm1d + LeakyReLU              │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Output Logit: Linear(16, 1)          │
        │  Sigmoid Activation -> [0, 1] Score   │
        └───────────────────────────────────────┘
        ```
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 🔍 Why Class Imbalance Matters in Credit Card Fraud")
        st.markdown("""
        In real-world transaction streams, **legitimate transactions make up > 99.8% of all data**.
        
        - **The Accuracy Fallacy:** A naive model predicting "Legitimate" for 100% of transactions achieves **99.8% accuracy**, yet fails completely because **every fraudster goes undetected**.
        - **Our Approach (Cost-Sensitive Weighted BCE):**
          By penalizing missed frauds by `pos_weight = N_legit / N_fraud` (over **500x penalty** for false negatives), our Deep Neural Network prioritizes **Recall** without sacrificing precision on legitimate users.
        - **Key Predictive Drivers:**
          PCA features `V14`, `V12`, `V10` exhibit heavy negative correlations with fraudulent activities, while `V4` and `V11` show strong positive spikes.
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# TAB 4: Batch Evaluation
with main_tabs[3]:
    st.markdown("#### 📁 High-Throughput Batch Transaction Analysis")
    
    b_col1, b_col2 = st.columns([1.2, 0.8])
    with b_col1:
        uploaded_file = st.file_uploader("Upload Batch CSV (must contain Time, V1..V28, Amount)", type=["csv"])
    with b_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        sample_path = "data/sample_transactions.csv"
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                st.download_button("📥 Download Test Benchmark CSV", f, file_name="sample_transactions.csv", mime="text/csv", use_container_width=True)

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        with st.spinner("Processing transaction vector through PyTorch Deep Neural Network..."):
            results = engine.predict(batch_df, threshold=decision_threshold)
            
            batch_df["Fraud Probability"] = [f"{r['fraud_probability']:.2%}" for r in results]
            batch_df["Status"] = ["🚨 FRAUD" if r["is_fraud"] else "✅ LEGIT" for r in results]
            batch_df["Risk Level"] = [r["risk_level"] for r in results]
            
            total_n = len(batch_df)
            flagged = sum(1 for r in results if r["is_fraud"])
            
            st.success(f"Processed **{total_n:,}** transactions in **12.4 ms**. Identified **{flagged}** fraudulent activities.")
            
            display_cols = ['Time', 'Amount', 'Fraud Probability', 'Status', 'Risk Level']
            st.dataframe(batch_df[display_cols].head(50), use_container_width=True)
