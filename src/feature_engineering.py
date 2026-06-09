from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


# Constantes — colunas manipuladas pelo pipeline


BINARY_COLS: list[str] = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]

DUMMY_COLS: list[str] = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

NUMERIC_COLS: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]



# Funções de transformação


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
   
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
   
    return df.reindex(columns=columns, fill_value=0)


def scale_numeric(
    df: pd.DataFrame,
    scaler: StandardScaler,
    fit: bool = False,
) -> tuple[pd.DataFrame, StandardScaler]:
    
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
   
    df = pd.DataFrame([row])

    # TotalCharges pode vir como string em dados brutos
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df = encode_features(df)
    df = align_columns(df, columns)
    df, _ = scale_numeric(df, scaler, fit=False)

    return df
