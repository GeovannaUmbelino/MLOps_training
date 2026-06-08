"""
main.py
=======
Ponto de entrada do pipeline de treinamento MLOps.

Execução:
    python main.py

O que acontece:
    1. Carrega os dados brutos
    2. Persiste uma cópia dos dados ORIGINAIS (antes de transformar)
    3. Pré-processa e salva scaler.pkl + columns.pkl
    4. Treina 6 modelos com rastreamento MLflow
    5. Salva best_model.pkl
    6. Plota curvas ROC comparativas
"""

from src.config import DATA_PATH, PROCESSED_DATA_PATH
from src.data_preprocessing import load_data, log_processed_data, preprocess_data
from src.model import plot_roc_curves, split_data, train_all_models


def main() -> None:
    print("📦 Carregando dados...")
    df = load_data(DATA_PATH)

    # Salva ANTES de transformar — assim o CSV processed tem os dados originais
    print("💾 Salvando cópia dos dados originais...")
    log_processed_data(df, PROCESSED_DATA_PATH)

    print("⚙️  Pré-processando dados...")
    X, y = preprocess_data(df)

    print("✂️  Dividindo em treino e teste...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n🔬 Treinando todos os modelos com MLflow tracking...")
    results = train_all_models(X_train, X_test, y_train, y_test)

    print("\n📊 Plotando curvas ROC...")
    plot_roc_curves(results, X_test, y_test)

    print("\n✅ Pipeline concluído!")
    print("   Explore os experimentos: mlflow ui")
    print("   Inicie a API:            uvicorn src.api:app --reload")
    print("   Inicie o frontend:       streamlit run src/app.py")


if __name__ == "__main__":
    main()
