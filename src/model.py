from __future__ import annotations

import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import MODEL_DIR, OUTPUT_DIR

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_data(
    X: "pd.DataFrame",
    y: "pd.Series",
) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.Series", "pd.Series"]:
    """Divide os dados em treino (80%) e teste (20%) com estratificação."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def _save_confusion_matrix(
    y_test: "pd.Series",
    y_pred,
    name: str,
) -> str:
    """Plota e salva a matriz de confusão; retorna o caminho do arquivo."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{name} — Confusion Matrix")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}_confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

def train_all_models(
    X_train: "pd.DataFrame",
    X_test: "pd.DataFrame",
    y_train: "pd.Series",
    y_test: "pd.Series",
) -> dict[str, dict]:
    """
    Treina 6 modelos, rastreia cada run no MLflow e salva o melhor.

    Para cada modelo são registrados:
        - Parâmetros: nome e hiperparâmetros do estimador
        - Métricas:   accuracy, f1, precision, recall, roc_auc
        - Artefatos:  matriz de confusão (PNG) + modelo serializado

    O melhor modelo (maior ROC AUC) é salvo em models/best_model.pkl
    e registrado numa run de sumário separada.

    Args:
        X_train, X_test: Features de treino e teste.
        y_train, y_test: Targets de treino e teste.

    Returns:
        Dict {nome_modelo: {"model": ..., "accuracy": ..., ...}}
    """
    candidates: dict[str, object] = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        # use_label_encoder removido — parâmetro descontinuado no XGBoost ≥ 1.6
        "XGBoost": XGBClassifier(eval_metric="logloss", verbosity=0),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    }

    best_model = None
    best_model_name: str | None = None
    best_score = 0.0
    results: dict[str, dict] = {}

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mlflow.set_experiment("churn_prediction")

    for name, model in candidates.items():
        print(f"\n🚀 Training {name}...")

        with mlflow.start_run(run_name=name):
            # --- Treino ---
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )

            # --- Métricas ---
            acc       = accuracy_score(y_test, y_pred)
            f1        = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall    = recall_score(y_test, y_pred)
            roc_auc   = roc_auc_score(y_test, y_proba) if y_proba is not None else None

            # --- Log MLflow ---
            mlflow.log_param("model_name", name)
            mlflow.log_params(model.get_params())

            mlflow.log_metric("accuracy",  acc)
            mlflow.log_metric("f1_score",  f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall",    recall)
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", roc_auc)

            # Artefato: matriz de confusão
            cm_path = _save_confusion_matrix(y_test, y_pred, name)
            mlflow.log_artifact(cm_path)

            # Artefato: modelo
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            # --- Resultado ---
            roc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
            print(f"  ✅ {name}: acc={acc:.4f} | f1={f1:.4f} | roc_auc={roc_str}")

            results[name] = {
                "model":     model,
                "accuracy":  acc,
                "f1_score":  f1,
                "precision": precision,
                "recall":    recall,
                "roc_auc":   roc_auc,
            }

            # --- Seleciona melhor ---
            score = roc_auc if roc_auc is not None else acc
            if score > best_score:
                best_score      = score
                best_model      = model
                best_model_name = name

    # --- Persiste o melhor modelo ---
    if best_model is not None:
        best_path = os.path.join(MODEL_DIR, "best_model.pkl")
        joblib.dump(best_model, best_path)
        print(f"\n🏆 Best model: {best_model_name} (roc_auc={best_score:.4f}) → {best_path}")

        with mlflow.start_run(run_name="best_model_summary"):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_roc_auc", best_score)
            mlflow.log_artifact(best_path)

    return results


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str) -> str:
    """
    Imprime o classification report e salva a matriz de confusão.

    Returns:
        Caminho da imagem gerada.
    """
    from sklearn.metrics import classification_report

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\nEvaluation for {model_name}")
    print(classification_report(y_test, y_pred))
    if y_proba is not None:
        print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

    return _save_confusion_matrix(y_test, y_pred, model_name)


def plot_roc_curves(models_dict: dict[str, dict], X_test, y_test) -> None:
    """
    Gera e salva a curva ROC comparativa de todos os modelos treinados.

    Args:
        models_dict: Dicionário retornado por train_all_models.
        X_test, y_test: Dados de avaliação.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, info in models_dict.items():
        model = info["model"] if isinstance(info, dict) else info

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc      = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC Curve — Model Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    roc_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 ROC curves saved to {roc_path}")
