import streamlit as st
import pandas as pd
import joblib
import time
from PIL import Image
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Security Portal | Thayss Fraud Guard",
    page_icon="🔒",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS INJECTION (BRAND IDENTITY) ---
st.markdown("""
    <style>
    /* Headers style */
    h1, h2, h3 {
        color: #002A8F !important;
        font-family: 'Arial', sans-serif;
    }
    /* Main button style */
    div.stButton > button:first-child {
        background-color: #EC7000;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #CC6000;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    /* Corporate divider */
    hr {
        border-top: 3px solid #EC7000;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # Ensure this file is in the same directory
        return joblib.load("fraud_detection_pipeline.pkl")
    except FileNotFoundError:
        return None

model = load_model()

# --- 4. PERSONAL LOGO ---
# Define your image filename
IMAGE_FILENAME = "logo c.png"

# Check if image exists before loading
if os.path.exists(IMAGE_FILENAME):
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        image = Image.open(IMAGE_FILENAME)
        st.image(image, use_container_width=True)
else:
    st.error(f"⚠️ Image file '{IMAGE_FILENAME}' not found in the current directory.")

# --- 5. APPLICATION HEADER ---
st.markdown("<h2 style='text-align: center;'>Fraud Detection System</h2>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; color: gray;'>
    <b>Powered by Advanced Machine Learning</b> | Enter the transaction details below to evaluate the risk level in real-time.
</p>
""", unsafe_allow_html=True)
st.divider()

# --- 6. USER INTERFACE (COLUMNS) ---
st.subheader("Transaction Data")
col_type, col_amount = st.columns(2)

with col_type:
    transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])
with col_amount:
    # Adjusted ranges to match the original dataset's high monetary magnitudes
    amount = st.number_input("Amount (Monetary Units)", min_value=0.0, max_value=100000000.0, value=150000.0, step=1000.0)

st.write("") # Blank space

col_orig, col_dest = st.columns(2)

with col_orig:
    st.markdown("#### 👤 Origin Account")
    oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, max_value=100000000.0, value=150000.0, step=1000.0)
    newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, max_value=100000000.0, value=0.0, step=1000.0)

with col_dest:
    st.markdown("#### 🏦 Destination Account")
    oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, max_value=100000000.0, value=0.0, step=1000.0)
    newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, max_value=100000000.0, value=150000.0, step=1000.0)

st.write("")
st.write("")

# --- 7. PREDICTION LOGIC ---
if st.button("Analyze Transaction"):
    if model is None:
        st.error("⚠️ Critical Error: Model file ('fraud_detection_pipeline.pkl') not found.")
    else:
        with st.spinner("Running security validation algorithms..."):
            time.sleep(1.5)
            
            input_data = pd.DataFrame([{
                "type" : transaction_type,
                "amount" : amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest
            }])
            
            prediction = model.predict(input_data)[0]
            
            st.divider()
            
            if prediction == 1:
                st.error("🚨 **SECURITY ALERT: HIGH RISK**")
                st.markdown("""
                This transaction's patterns match known fraudulent behaviors. 
                * **Suggested Action:** Preventative block and manual review by an analyst.
                """)
            else:
                st.success("✅ **SAFE OPERATION: LOW RISK**")
                st.markdown("""
                The transaction meets normal operational security parameters.
                * **Suggested Action:** Approve processing.
                """)
