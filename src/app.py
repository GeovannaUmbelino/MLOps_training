from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Garante que `src` é importável quando chamado de qualquer diretório
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_DIR
from src.feature_engineering import preprocess_inference_row


# Configuração da página

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
)



# Carregamento de artefatos (cacheado para não recarregar a cada interação)


@st.cache_resource
def load_artefacts():
    model  = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    cols   = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))
    return model, scaler, cols



# Visualização — gauge de probabilidade


def _gauge_chart(probability: float):
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.15, 1.2)
    ax.axis("off")

    theta_bg   = np.linspace(np.pi, 0, 200)
    theta_fill = np.linspace(np.pi, np.pi - probability * np.pi, 200)

    ax.plot(np.cos(theta_bg), np.sin(theta_bg),
            color="#e0e0e0", linewidth=18, solid_capstyle="round")

    color = "#e74c3c" if probability > 0.5 else "#2ecc71"
    ax.plot(np.cos(theta_fill), np.sin(theta_fill),
            color=color, linewidth=18, solid_capstyle="round")

    angle = np.pi - probability * np.pi
    ax.annotate(
        "",
        xy=(0.65 * np.cos(angle), 0.65 * np.sin(angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2),
    )

    ax.text(0, -0.08, f"{probability * 100:.1f}%",
            ha="center", va="center", fontsize=22, fontweight="bold")
    ax.text(-1.1, -0.08, "0%",   ha="center", fontsize=10, color="grey")
    ax.text(1.1,  -0.08, "100%", ha="center", fontsize=10, color="grey")
    ax.set_title("Probabilidade de Churn", fontsize=13, fontweight="bold", pad=12)
    return fig



# UI principal


st.title("📡 Telco Customer Churn Predictor")
st.markdown("Preencha os dados do cliente abaixo e clique em **Predict** para ver o risco de churn.")

# Tenta carregar artefatos locais
try:
    model, scaler, columns = load_artefacts()
    artefacts_ok = True
except Exception as e:
    st.error(f"Não foi possível carregar os artefatos do modelo: {e}")
    artefacts_ok = False

# Sidebar
st.sidebar.header("⚙️ Configurações")
use_api = st.sidebar.toggle("Usar backend FastAPI", value=False)
api_url = st.sidebar.text_input("URL da API", value="http://localhost:8000")
if use_api:
    st.sidebar.info("A API precisa estar rodando:\n`uvicorn src.api:app --reload`")


# Formulário

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Dados Pessoais")
        gender     = st.selectbox("Gênero", ["Male", "Female"])
        senior     = st.selectbox("Idoso", [0, 1], format_func=lambda x: "Sim" if x else "Não")
        partner    = st.selectbox("Tem cônjuge", ["Yes", "No"])
        dependents = st.selectbox("Tem dependentes", ["Yes", "No"])

    with col2:
        st.subheader("📱 Serviços")
        phone_service  = st.selectbox("Serviço de telefone", ["Yes", "No"])
        multiple_lines = st.selectbox("Múltiplas linhas", ["No", "Yes", "No phone service"])
        internet       = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
        online_sec     = st.selectbox("Segurança online", ["No", "Yes", "No internet service"])
        online_bkp     = st.selectbox("Backup online", ["No", "Yes", "No internet service"])
        device_prot    = st.selectbox("Proteção de dispositivo", ["No", "Yes", "No internet service"])
        tech_support   = st.selectbox("Suporte técnico", ["No", "Yes", "No internet service"])
        streaming_tv   = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_mv   = st.selectbox("Streaming filmes", ["No", "Yes", "No internet service"])

    with col3:
        st.subheader("💳 Contrato e Cobrança")
        tenure   = st.slider("Tempo como cliente (meses)", 0, 72, 12)
        contract = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Fatura digital", ["Yes", "No"])
        payment  = st.selectbox("Método de pagamento", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ])
        monthly = st.slider("Cobrança mensal (US$)", 18.0, 120.0, 50.0, step=0.5)
        default_total = float(monthly * tenure) if tenure > 0 else 50.0
        total = st.number_input("Cobrança total (US$)", min_value=0.0, value=default_total, step=10.0)

    submitted = st.form_submit_button("🔍 Prever Churn", use_container_width=True)



# Predição

if submitted:
    customer = {
        "gender": gender, "SeniorCitizen": senior,
        "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet,
        "OnlineSecurity": online_sec, "OnlineBackup": online_bkp,
        "DeviceProtection": device_prot, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_mv,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }

    churn, probability, model_name = False, 0.0, "N/A"

    if use_api:
        try:
            import requests
            resp = requests.post(f"{api_url}/predict", json=customer, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            churn       = data["churn"]
            probability = data["churn_probability"]
            model_name  = data["model_used"]
        except Exception as e:
            st.error(f"Falha na chamada à API: {e}. Usando inferência local.")
            use_api = False

    if not use_api:
        if artefacts_ok:
            # Usa a mesma função de feature_engineering — sem duplicar lógica
            X = preprocess_inference_row(
                row=customer.copy(),
                scaler=scaler,
                columns=columns,
            )
            churn      = bool(model.predict(X)[0])
            probability = (
                float(model.predict_proba(X)[0][1])
                if hasattr(model, "predict_proba")
                else float(churn)
            )
            model_name = type(model).__name__
        else:
            st.error("Artefatos locais não disponíveis. Inicie a API ou execute `python main.py`.")
            st.stop()

    # --- Resultado ---
    st.divider()
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if churn:
            st.error("## ⚠️ ALTO RISCO DE CHURN\nEste cliente provavelmente vai cancelar.")
        else:
            st.success("## ✅ BAIXO RISCO DE CHURN\nEste cliente provavelmente vai continuar.")

        st.markdown(f"**Modelo utilizado:** `{model_name}`")
        st.markdown(f"**Probabilidade de churn:** `{probability * 100:.1f}%`")

    with res_col2:
        fig = _gauge_chart(probability)
        st.pyplot(fig, use_container_width=True)

    # Resumo dos inputs
    with st.expander("📋 Dados inseridos"):
        display = {k: [v] for k, v in customer.items()}
        st.dataframe(pd.DataFrame(display).T.rename(columns={0: "Valor"}))
