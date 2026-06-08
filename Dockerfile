FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    matplotlib mlflow numpy pandas scikit-learn seaborn xgboost \
    fastapi "uvicorn[standard]" streamlit joblib pydantic requests

# Copy source
COPY . .

EXPOSE 8000 8501 5000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
