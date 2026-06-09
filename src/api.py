from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import MODEL_DIR
from src.feature_engineering import preprocess_inference_row


# Schemas Pydantic


class CustomerFeatures(BaseModel):
    gender:           Literal["Male", "Female"]
    SeniorCitizen:    Literal[0, 1]
    Partner:          Literal["Yes", "No"]
    Dependents:       Literal["Yes", "No"]
    tenure:           float = Field(..., ge=0, le=100)
    PhoneService:     Literal["Yes", "No"]
    MultipleLines:    Literal["No", "Yes", "No phone service"]
    InternetService:  Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity:   Literal["No", "Yes", "No internet service"]
    OnlineBackup:     Literal["No", "Yes", "No internet service"]
    DeviceProtection: Literal["No", "Yes", "No internet service"]
    TechSupport:      Literal["No", "Yes", "No internet service"]
    StreamingTV:      Literal["No", "Yes", "No internet service"]
    StreamingMovies:  Literal["No", "Yes", "No internet service"]
    Contract:         Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod:    Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges:   float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "TotalCharges": 29.85,
            }
        }
    }


class PredictionResponse(BaseModel):
    churn:             bool
    churn_label:       str
    churn_probability: float
    model_used:        str



# Estado compartilhado entre requests (carregado uma vez no startup)


state: dict[str, Any] = {}



# Lifespan — carrega artefatos uma única vez ao subir a API


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path   = os.path.join(MODEL_DIR, "best_model.pkl")
    scaler_path  = os.path.join(MODEL_DIR, "scaler.pkl")
    columns_path = os.path.join(MODEL_DIR, "columns.pkl")

    missing = [p for p in (model_path, scaler_path, columns_path) if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"Artefatos não encontrados: {missing}. Execute `python main.py` primeiro."
        )

    state["model"]      = joblib.load(model_path)
    state["scaler"]     = joblib.load(scaler_path)
    state["columns"]    = joblib.load(columns_path)
    state["model_name"] = type(state["model"]).__name__
    print(f"✅ Modelo carregado: {state['model_name']} | {len(state['columns'])} features")
    yield
    state.clear()



# Aplicação


app = FastAPI(
    title="Churn Prediction API",
    description="Prediz probabilidade de churn de clientes de telecomunicações.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# Endpoints


@app.get("/", tags=["Root"])
def root() -> dict:
    return {"message": "Churn Prediction API", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["Monitoring"])
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info", tags=["Monitoring"])
def model_info() -> dict:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    return {
        "model_type": state["model_name"],
        "n_features": len(state["columns"]),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    try:
        X = preprocess_inference_row(
            row=customer.model_dump(),
            scaler=state["scaler"],
            columns=state["columns"],
        )
        model = state["model"]
        pred  = int(model.predict(X)[0])
        proba = (
            float(model.predict_proba(X)[0][1])
            if hasattr(model, "predict_proba")
            else float(pred)
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PredictionResponse(
        churn=bool(pred),
        churn_label="Churn" if pred else "No Churn",
        churn_probability=round(proba, 4),
        model_used=state["model_name"],
    )
