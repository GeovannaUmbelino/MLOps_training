"""
src/data_preprocessing.py
==========================
Responsável por carregar, limpar e preparar os dados brutos para o treino.
Usa feature_engineering.py para encoding e scaling — sem duplicar lógica.
"""

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
    """
    Limpa e transforma o DataFrame bruto para uso no treino.

    IMPORTANTE: nunca modifica o DataFrame original — trabalha em uma cópia.

    Etapas:
        1. Remove customerID (identificador sem valor preditivo)
        2. Corrige TotalCharges (pode vir como string com espaços)
        3. Encoding via feature_engineering.encode_features
        4. Scaling das features numéricas + persistência do scaler e colunas
        5. Separação em X (features) e y (target)

    Args:
        df: DataFrame bruto carregado de churn-data.csv.

    Returns:
        Tupla (X, y) prontos para train_test_split.
    """
    # --- Cópia: nunca altera o DataFrame do chamador ---
    df = df.copy()

    # 1. Remove identificador
    df = df.drop(columns=["customerID"], errors="ignore")

    # 2. Corrige TotalCharges (espaços viram NaN, preenchemos com mediana)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 3. Encoding (inclui mapeamento Churn Yes/No → 1/0 via binary_cols)
    #    Churn precisa ser mapeado manualmente antes do encode_features
    #    porque não está na lista BINARY_COLS (é o target, não feature)
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
    """
    Salva o DataFrame bruto (com customerID e Churn ainda presentes)
    como CSV processado para auditoria.

    Deve ser chamado com o DataFrame ANTES de preprocess_data,
    ou com uma cópia limpa do original.

    Args:
        df:          DataFrame original (não transformado).
        output_path: Caminho de saída do CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data logged to {output_path}")
