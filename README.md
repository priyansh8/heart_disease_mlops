# Heart Disease Risk Classifier — End-to-End MLOps Project

This repo implements an end-to-end MLOps workflow for the **Heart Disease UCI** dataset:
- data download + cleaning
- EDA with saved plots
- feature engineering + model training (LogReg + RandomForest)
- **MLflow** experiment tracking
- reproducible preprocessing pipeline
- unit tests + linting + GitHub Actions CI
- Dockerized **FastAPI** inference service (`/predict`, `/health`, `/metrics`)
- Kubernetes manifests for local deployment (Minikube / Docker Desktop)
- logging + basic Prometheus-compatible metrics
- a docx report template (fill in screenshots after you run)

> Note: Deployment screenshots and “deployed URL” depend on your environment. This repo includes **everything needed to run and generate** those outputs; you will need to capture screenshots after running the commands.

## 1) Quickstart (local)

### 1.1 Create venv & install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### 1.2 Download data
```bash
python scripts/download_data.py --out data/raw/heart.csv
```

### 1.3 Run EDA (saves plots into `reports/figures/`)
```bash
python scripts/run_eda.py --data data/raw/heart.csv --outdir reports/figures
```

### 1.4 Train models (MLflow logs to `mlruns/`)
```bash
python -m src.train --data data/raw/heart.csv --outdir artifacts --experiment heart_disease --seed 42
```

The trained pipeline is saved to:
- `artifacts/model.joblib` (used by API)
- MLflow run artifacts under `mlruns/`

### 1.5 Run API locally
```bash
export MODEL_PATH=artifacts/model.joblib  # Windows Powershell: $env:MODEL_PATH="artifacts/model.joblib"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Test:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @scripts/sample_request.json
```

Open metrics:
- http://localhost:8000/metrics

## 2) Docker

### 2.1 Build
```bash
docker build -t heart-api:latest .
```

### 2.2 Run (mount model)
```bash
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=/app/artifacts/model.joblib \
  -v $(pwd)/artifacts:/app/artifacts \
  heart-api:latest
```

## 3) Kubernetes (Minikube)

1. Build image and load into minikube:
```bash
minikube start
eval $(minikube docker-env)
docker build -t heart-api:latest .
```

2. Create namespace + deploy:
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -n mlops-heart -f k8s/
```

3. Port-forward:
```bash
kubectl -n mlops-heart port-forward svc/heart-api 8000:80
```

Then:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @scripts/sample_request.json
```

## 4) MLflow UI
```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

## 5) Tests & Lint
```bash
ruff check .
black --check .
pytest -q
```

## Repo Structure

```
.
├── artifacts/                  # generated outputs (model.joblib, metrics.json)
├── data/
│   ├── raw/                    # downloaded raw CSV
│   └── processed/              # optional processed outputs
├── docs/
│   └── report.docx             # 10-page report template (fill screenshots)
├── k8s/                        # Kubernetes manifests
├── reports/
│   └── figures/                # EDA plots
├── scripts/                    # CLI scripts
├── src/
│   ├── api/                    # FastAPI service
│   └── ...                     # training + data modules
└── tests/                      # pytest unit tests
```

## Inputs

The API expects UCI “Cleveland processed” schema fields:

- age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

Example request is at `scripts/sample_request.json`.

## License
Educational use.
