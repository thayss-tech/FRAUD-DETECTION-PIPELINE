import streamlit as st
import pandas as pd
import joblib
import time
# Para cargar imágenes locales necesitamos esta librería adicional (ya viene con Python)
from PIL import Image
import os

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Portal de Segurança | Itaú",
    page_icon="🔒",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. INYECCIÓN DE CSS (IDENTIDAD ITAÚ) ---
st.markdown("""
    <style>
    /* Estilo para los títulos */
    h1, h2, h3 {
        color: #002A8F !important;
        font-family: 'Arial', sans-serif;
    }
    /* Estilo del botón principal */
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
    /* Separador corporativo */
    hr {
        border-top: 3px solid #EC7000;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. CARGAR EL MODELO ---
@st.cache_resource
def load_model():
    try:
        # Asegúrate de que este archivo también esté en la misma carpeta
        return joblib.load("fraud_detection_pipeline.pkl")
    except FileNotFoundError:
        return None

model = load_model()

# --- 4. LOGÓTIPO DO ITAÚ LOCAL ---
# Definimos el nombre de tu archivo de imagen
IMAGE_FILENAME = "logo_itau.png"

# Verificamos si la imagen existe antes de intentar cargarla
if os.path.exists(IMAGE_FILENAME):
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        image = Image.open(IMAGE_FILENAME)
        st.image(image, use_container_width=True)
else:
    st.error(f"⚠️ No se encontró el archivo de imagen '{IMAGE_FILENAME}' en la carpeta actual.")

# --- 5. CABEÇALHO DA APLICAÇÃO ---
st.markdown("<h2 style='text-align: center;'>Sistema de Deteção de Fraude</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><b>Centro de Operações de Segurança (SOC)</b> | Por favor, insira os detalhes da transação para avaliar o nível de risco.</p>", unsafe_allow_html=True)
st.divider()

# --- 6. INTERFACE DE UTILIZADOR (COLUNAS) ---
st.subheader("Dados da Transação")
col_type, col_amount = st.columns(2)

with col_type:
    transaction_type = st.selectbox("Tipo de Operação", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])
with col_amount:
    amount = st.number_input("Montante (BRL)", min_value=0.0, value=1000.0, step=100.0)

st.write("") # Espaço em branco

col_orig, col_dest = st.columns(2)

with col_orig:
    st.markdown("#### 👤 Conta de Origem")
    oldbalanceOrg = st.number_input("Saldo Anterior (Origem)", min_value=0.0, value=1000.0, step=500.0)
    newbalanceOrig = st.number_input("Novo Saldo (Origem)", min_value=0.0, value=0.0, step=500.0)

with col_dest:
    st.markdown("#### 🏦 Conta de Destino")
    oldbalanceDest = st.number_input("Saldo Anterior (Destino)", min_value=0.0, value=0.0, step=500.0)
    newbalanceDest = st.number_input("Novo Saldo (Destino)", min_value=0.0, value=1000.0, step=500.0)

st.write("")
st.write("")

# --- 7. LÓGICA DE PREVISÃO ---
if st.button("Analisar Transação"):
    if model is None:
        st.error("⚠️ Erro crítico: Não foi encontrado o ficheiro do modelo ('fraud_detection_pipeline.pkl').")
    else:
        with st.spinner("A executar algoritmos de validação de segurança..."):
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
                st.error("🚨 **ALERTA DE SEGURANÇA: ALTO RISCO**")
                st.markdown("""
                Os padrões desta operação coincidem com comportamentos fraudulentos. 
                * **Ação sugerida:** Bloqueio preventivo da transação e revisão manual por um analista.
                """)
            else:
                st.success("✅ **OPERAÇÃO SEGURA: BAIXO RISCO**")
                st.markdown("""
                A transação cumpre com os parâmetros normais de segurança operativa.
                * **Ação sugerida:** Aprovar processamento.
                """)