from __future__ import annotations

import os

import joblib
import pandas as pd

from src.config import MODEL_DIR
from src.feature_engineering import (
    NUMERIC_COLS,
    encode_features,
    scale_numeric,
)


def load_data(path: str) -> pd.DataFrame:
    """Carrega o CSV bruto e retorna um DataFrame."""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
   
    # --- Cópia: nunca altera o DataFrame do chamador ---
    df = df.copy()

    # 1. Remove identificador
    df = df.drop(columns=["customerID"], errors="ignore")

    # 2. Corrige TotalCharges (espaços viram NaN, preenchemos com mediana)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 3. Encoding (inclui mapeamento Churn Yes/No → 1/0 via binary_cols)
  
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df = encode_features(df)

    # 4. Separar target antes do scaling
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # 5. Scaling + persistência dos artefatos para inferência
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X, scaler = scale_numeric(X, scaler, fit=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "columns.pkl"))

    return X, y


def log_processed_data(
    df: pd.DataFrame,
    output_path: str = "data/processed/churn-processed.csv",
) -> None:
   
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data logged to {output_path}")
