# 📡 Telco Customer Churn — Pipeline MLOps

Pipeline de machine learning **end-to-end** para prever o churn (cancelamento) de clientes de uma empresa de telecomunicações fictícia na Califórnia. O projeto cobre desde a exploração dos dados até o deploy do modelo em produção, com rastreamento de experimentos, API de inferência e interface gráfica.

---

## 🎯 Objetivo

Dado o [dataset Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) com **7.043 clientes** e **19 features** (dados demográficos, serviços contratados e faturamento), construir um pipeline completo que:

- Treina e compara **6 modelos** de classificação
- Rastreia todos os experimentos com **MLflow**
- Disponibiliza o melhor modelo via **API REST (FastAPI)**
- Permite interação visual via **interface Streamlit**
- É totalmente contêinerizado com **Docker**

---

## 📁 Estrutura do Projeto

```
MLOps_training/
│
├── data/
│   ├── raw/                        # Dataset original
│   └── processed/                  # Dataset antes das transformações
│
├── models/
│   ├── best_model.pkl             
│   ├── scaler.pkl                  # StandardScaler ajustado no treino
│   └── columns.pkl                 # Features do treino (alinhamento inferência)
│
├── notebooks/
│   └── EDA.ipynb                   # Análise exploratória de dados
│
├── outputs/                        # Matrizes de confusão + curva ROC
│
├── src/
│   ├── config.py                   # Caminhos e configurações globais
│   ├── feature_engineering.py      # Fonte única de transformações de features
│   ├── data_preprocessing.py       # Carga, limpeza, encoding e scaling
│   ├── model.py                    # Treinamento com MLflow tracking
│   ├── visualization.py            # Plots auxiliares
│   ├── api.py                      # API FastAPI de inferência
│   └── app.py                      # Interface Streamlit
│
├── main.py                         # Ponto de entrada do pipeline
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 🤖 Modelos Treinados

| Modelo | Métricas rastreadas |
|---|---|
| Logistic Regression | accuracy, f1, precision, recall, roc_auc |
| Random Forest | idem |
| XGBoost | idem |
| KNN | idem |
| SVM | idem |
| MLP (Neural Network) | idem |

O melhor modelo é selecionado automaticamente por **ROC AUC** e salvo em `models/best_model.pkl`.

---

## 🚀 Como Rodar

### Pré-requisito

Ter o dataset `churn-data.csv` em `data/raw/`. Download: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

### Opção 1 — Local

```bash
# 1. Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Instalar dependências
pip install mlflow fastapi "uvicorn[standard]" streamlit xgboost \
            scikit-learn pandas numpy joblib pydantic requests \
            matplotlib seaborn

# 3. Treinar os modelos (gera best_model.pkl, scaler.pkl, columns.pkl)
python main.py

# 4. Explorar experimentos no MLflow  →  http://localhost:5000
mlflow ui

# 5. Subir a API FastAPI  →  http://localhost:8000/docs
uvicorn src.api:app --reload

# 6. Subir a interface Streamlit  →  http://localhost:8501
streamlit run src/app.py
```

> Os passos 4, 5 e 6 rodam em **terminais separados** simultaneamente.

---

### Opção 2 — Docker Compose

```bash
# Subir MLflow + API + Streamlit
docker-compose up --build

# Em outro terminal: treinar os modelos dentro do container
docker-compose run --rm train
```

| Serviço | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| FastAPI (Swagger) | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

---

## 📬 Exemplo de Requisição à API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

**Resposta:**
```json
{
  "churn": true,
  "churn_label": "Churn",
  "churn_probability": 0.7312,
  "model_used": "LogisticRegression"
}
```



---

## 🛠️ Stack

| Categoria | Tecnologia |
|---|---|
| Linguagem | Python 3.12 |
| Dados | Pandas, NumPy |
| Modelos | scikit-learn, XGBoost |
| Rastreamento | MLflow |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit, Matplotlib |
| Deploy | Docker, Docker Compose |
| Visualizações | Matplotlib, Seaborn |

---

## 📄 Licença

Projeto desenvolvido para fins educacionais — Desafio Trenne.
