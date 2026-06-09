from src.config import DATA_PATH, PROCESSED_DATA_PATH
from src.data_preprocessing import load_data, log_processed_data, preprocess_data
from src.model import plot_roc_curves, split_data, train_all_models


def main() -> None:
    print(" Carregando dados...")
    df = load_data(DATA_PATH)

    # Salva ANTES de transformar — assim o CSV processed tem os dados originais
    print(" Salvando cópia dos dados originais...")
    log_processed_data(df, PROCESSED_DATA_PATH)

    print("  Pré-processando dados...")
    X, y = preprocess_data(df)

    print(" Dividindo em treino e teste...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n Treinando todos os modelos com MLflow tracking...")
    results = train_all_models(X_train, X_test, y_train, y_test)

    print("\n Plotando curvas ROC...")
    plot_roc_curves(results, X_test, y_test)

    print("\n Pipeline concluído!")
    print("   Explore os experimentos: mlflow ui")
    print("   Inicie a API:            uvicorn src.api:app --reload")
    print("   Inicie o frontend:       streamlit run src/app.py")


if __name__ == "__main__":
    main()
