"""
Interactive Streamlit Web Application:
Credit Card Fraud Detection using Deep Neural Network
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch

from src.models import FraudDNN
from src.predict import FraudPredictor

# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection | Deep Neural Network",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e222d;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .fraud-alert {
        background-color: rgba(239, 83, 80, 0.15);
        border: 1px solid #ef5350;
        padding: 1rem;
        border-radius: 8px;
        color: #ef5350;
        font-weight: 600;
    }
    .legit-alert {
        background-color: rgba(102, 187, 106, 0.15);
        border: 1px solid #66bb6a;
        padding: 1rem;
        border-radius: 8px;
        color: #66bb6a;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Cache model predictor
@st.cache_resource
def get_predictor():
    if not os.path.exists("models/best_dnn_model.pth") or not os.path.exists("models/preprocessors.joblib"):
        return None
    try:
        return FraudPredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

predictor = get_predictor()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9521/9521255.png", width=64)
    st.title("Fraud Guard DNN")
    st.markdown("**Deep Neural Network** Inference Engine")
    
    st.divider()
    decision_threshold = st.slider(
        "Decision Threshold", 
        min_value=0.05, 
        max_value=0.95, 
        value=0.50, 
        step=0.05,
        help="Adjust classification threshold for flagging fraud."
    )
    
    st.markdown("### 🎛️ Sample Scenarios")
    sample_type = st.radio(
        "Quick Pre-load:",
        ["Custom Manual Input", "Typical Legitimate Transaction", "Suspicious / Anomaly Transaction", "High-Risk Fraud Transaction"]
    )
    
    st.divider()
    st.markdown("""
    **Model Specs:**
    - Architecture: 4-Layer Deep MLP
    - Regularization: BatchNorm + Dropout
    - Training: Weighted Loss (Pos Weight)
    - Target: Kaggle Imbalanced Dataset
    """)

# Main Screen
st.markdown('<div class="main-header">💳 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time fraud scoring powered by a PyTorch Deep Neural Network</div>', unsafe_allow_html=True)

if predictor is None:
    st.warning("⚠️ **Trained Model Artifacts Not Detected Yet**")
    st.info("Run the training script first by executing in your terminal:\n```bash\npython -m src.train\n```\nThis trains the Deep Neural Network locally and generates all required weight files.")
    st.stop()

# Preset feature values
preset_values = {f"V{i}": 0.0 for i in range(1, 29)}
preset_time = 3600.0
preset_amount = 75.00

if sample_type == "Typical Legitimate Transaction":
    preset_time = 14200.0
    preset_amount = 45.50
    preset_values["V14"] = 0.25
    preset_values["V4"] = -0.15
    preset_values["V11"] = -0.05
    preset_values["V12"] = 0.10
elif sample_type == "Suspicious / Anomaly Transaction":
    preset_time = 45000.0
    preset_amount = 950.00
    preset_values["V14"] = -1.8
    preset_values["V4"] = 1.6
    preset_values["V11"] = 1.2
    preset_values["V12"] = -1.5
elif sample_type == "High-Risk Fraud Transaction":
    preset_time = 72000.0
    preset_amount = 1450.00
    preset_values["V14"] = -4.5  # Heavy negative V14
    preset_values["V4"] = 3.9    # High positive V4
    preset_values["V11"] = 3.2   # High positive V11
    preset_values["V12"] = -3.8  # Heavy negative V12

tabs = st.tabs(["🔍 Live Transaction Tester", "📊 Model Performance & Reports", "📁 Batch Evaluation"])

with tabs[0]:
    col_left, col_right = st.columns([1.1, 0.9])
    
    with col_left:
        st.subheader("Transaction Parameters")
        c1, c2 = st.columns(2)
        with c1:
            time_val = st.number_input("Time (Seconds from first transaction)", value=float(preset_time), step=100.0)
        with c2:
            amount_val = st.number_input("Transaction Amount ($)", value=float(preset_amount), step=10.0, min_value=0.0)
            
        st.markdown("**Key PCA Features (Highest Correlation to Fraud):**")
        k1, k2 = st.columns(2)
        with k1:
            v14_val = st.slider("V14 (Strong negative correlation)", -10.0, 5.0, float(preset_values["V14"]), step=0.1)
            v12_val = st.slider("V12 (Negative correlation)", -10.0, 5.0, float(preset_values["V12"]), step=0.1)
        with k2:
            v4_val = st.slider("V4 (Strong positive correlation)", -5.0, 10.0, float(preset_values["V4"]), step=0.1)
            v11_val = st.slider("V11 (Positive correlation)", -5.0, 10.0, float(preset_values["V11"]), step=0.1)
            
        with st.expander("🛠️ Advanced: Adjust Remaining PCA Features (V1 - V28)"):
            grid_cols = st.columns(4)
            for i in range(1, 29):
                if f"V{i}" not in ["V14", "V4", "V11", "V12"]:
                    col_idx = (i - 1) % 4
                    with grid_cols[col_idx]:
                        preset_values[f"V{i}"] = st.number_input(f"V{i}", value=float(preset_values[f"V{i}"]), step=0.2)
                        
        preset_values["V14"] = v14_val
        preset_values["V4"] = v4_val
        preset_values["V11"] = v11_val
        preset_values["V12"] = v12_val
        
        # Build transaction payload
        tx_data = {'Time': time_val, 'Amount': amount_val}
        tx_data.update(preset_values)

    with col_right:
        st.subheader("Deep Neural Network Assessment")
        
        # Predict
        result = predictor.predict(tx_data, threshold=decision_threshold)
        prob = result["fraud_probability"]
        is_fraud = result["is_fraud"]
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Probability Index", 'font': {'size': 20, 'color': '#ffffff'}},
            number = {'suffix': "%", 'font': {'size': 32, 'color': '#ffffff'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#ef5350" if is_fraud else "#42a5f5"},
                'bgcolor': "rgba(0,0,0,0.2)",
                'borderwidth': 2,
                'bordercolor': "#444",
                'steps': [
                    {'range': [0, 25], 'color': 'rgba(102, 187, 106, 0.4)'},
                    {'range': [25, 60], 'color': 'rgba(255, 167, 38, 0.4)'},
                    {'range': [60, 100], 'color': 'rgba(239, 83, 80, 0.4)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': decision_threshold * 100
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
        if is_fraud:
            st.markdown(f"""
            <div class="fraud-alert">
                🚨 <b>FRAUD DETECTED</b><br>
                Model probability ({prob:.2%}) exceeds classification threshold ({decision_threshold:.0%}).
                Transaction flagged for immediate freeze and verification.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="legit-alert">
                ✅ <b>TRANSACTION APPROVED (LEGITIMATE)</b><br>
                Fraud risk ({prob:.2%}) is safely below the decision threshold.
                Transaction cleared for automated settlement.
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Fraud Probability", f"{prob:.4f}")
        with m2:
            st.metric("Model Confidence", f"{result['confidence_score']:.2%}")

with tabs[1]:
    st.subheader("Evaluation Metrics & Diagnostic Visualizations")
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        st.markdown("#### Confusion Matrix")
        if os.path.exists("reports/figures/confusion_matrix.png"):
            st.image("reports/figures/confusion_matrix.png", use_container_width=True)
        else:
            st.info("Run `python -m src.train` to generate confusion matrix visualization.")
            
    with f_col2:
        st.markdown("#### ROC & Precision-Recall Curves")
        if os.path.exists("reports/figures/roc_pr_curves.png"):
            st.image("reports/figures/roc_pr_curves.png", use_container_width=True)
        else:
            st.info("Run `python -m src.train` to generate ROC & PR curve visualizations.")
            
    st.markdown("#### Training & Validation Loss Progression")
    if os.path.exists("reports/figures/training_history.png"):
        st.image("reports/figures/training_history.png", use_container_width=True)

with tabs[2]:
    st.subheader("Batch Transaction Scoring")
    uploaded_file = st.file_uploader("Upload CSV containing transaction records", type=["csv"])
    
    sample_file_path = "data/sample_transactions.csv"
    if os.path.exists(sample_file_path):
        with open(sample_file_path, "rb") as f:
            st.download_button("📥 Download Sample Evaluation CSV", f, file_name="sample_transactions.csv", mime="text/csv")
            
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(batch_df)} transactions. Processing inference through Deep Neural Network...")
        results = predictor.predict(batch_df, threshold=decision_threshold)
        
        batch_df['fraud_probability'] = [r['fraud_probability'] for r in results]
        batch_df['prediction'] = ["FRAUD" if r['is_fraud'] else "LEGITIMATE" for r in results]
        
        st.dataframe(batch_df[['Time', 'Amount', 'fraud_probability', 'prediction']].head(20), use_container_width=True)
        
        fraud_count = sum(r['is_fraud'] for r in results)
        st.success(f"Batch completed: {fraud_count} flagged as Fraud out of {len(batch_df)} transactions.")
