"""
src/feature_engineering.py
===========================
Fonte única de verdade para todas as transformações de features do dataset
de churn. Tanto o pipeline de treino (data_preprocessing.py) quanto a
inferência (api.py, app.py) devem importar daqui — nunca duplicar.

Por que centralizar?
    Qualquer divergência entre a lógica de treino e a de inferência introduz
    "training-serving skew": o modelo recebe dados diferentes dos que aprendeu,
    degradando silenciosamente as predições sem levantar erros.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constantes — colunas manipuladas pelo pipeline
# ---------------------------------------------------------------------------

BINARY_COLS: list[str] = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]

DUMMY_COLS: list[str] = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

NUMERIC_COLS: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]


# ---------------------------------------------------------------------------
# Funções de transformação
# ---------------------------------------------------------------------------

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica encoding categórico ao DataFrame *sem modificar o original*.

    Etapas (ordem idêntica ao treino):
        1. Mapeamento binário Yes/No → 1/0
        2. Codificação de gênero Male/Female → 1/0
        3. One-hot encoding das colunas categóricas multi-classe

    Args:
        df: DataFrame com as colunas brutas do dataset (sem customerID/Churn).

    Returns:
        Novo DataFrame com as colunas codificadas.
    """
    df = df.copy()

    # 1. Mapeamento binário
    yes_no = {"Yes": 1, "No": 0}
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(yes_no)

    # 2. Gênero
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # 3. One-hot encoding
    existing_dummy_cols = [c for c in DUMMY_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=existing_dummy_cols)

    return df


def align_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Alinha o DataFrame às colunas do conjunto de treino.

    Colunas ausentes são preenchidas com 0 (categoria não vista).
    Colunas extras são descartadas. Isso garante que a inferência
    sempre receba exatamente o mesmo número e ordem de features.

    Args:
        df:      DataFrame já encodado.
        columns: Lista de colunas salva em columns.pkl durante o treino.

    Returns:
        DataFrame com exatamente as colunas de `columns`, na mesma ordem.
    """
    return df.reindex(columns=columns, fill_value=0)


def scale_numeric(
    df: pd.DataFrame,
    scaler: StandardScaler,
    fit: bool = False,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Escala as features numéricas com StandardScaler.

    Args:
        df:     DataFrame com NUMERIC_COLS presentes.
        scaler: Instância de StandardScaler.
        fit:    Se True, chama fit_transform (treino).
                Se False, chama apenas transform (inferência).

    Returns:
        Tupla (df_escalado, scaler).
    """
    df = df.copy()
    existing = [c for c in NUMERIC_COLS if c in df.columns]
    if fit:
        df[existing] = scaler.fit_transform(df[existing])
    else:
        df[existing] = scaler.transform(df[existing])
    return df, scaler


def preprocess_inference_row(
    row: dict,
    scaler: StandardScaler,
    columns: list[str],
) -> pd.DataFrame:
    """
    Transforma um dicionário de features brutas (vindo da API ou do Streamlit)
    em um DataFrame pronto para ser passado ao modelo.

    Aplica a sequência completa: encode → align → scale.

    Args:
        row:     Dict com os campos do cliente (os mesmos do dataset,
                 exceto customerID e Churn).
        scaler:  StandardScaler ajustado durante o treino (scaler.pkl).
        columns: Lista de features do treino (columns.pkl).

    Returns:
        DataFrame com 1 linha e exatamente as colunas do modelo.
    """
    df = pd.DataFrame([row])

    # TotalCharges pode vir como string em dados brutos
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df = encode_features(df)
    df = align_columns(df, columns)
    df, _ = scale_numeric(df, scaler, fit=False)

    return df
