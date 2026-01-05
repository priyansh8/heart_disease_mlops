# MLOps Assignment - Complete Commands Guide

## 📁 Project Structure
```
assignment1_mlops/
├── COMMANDS.md                    ← Quick reference
├── assignment1.ipynb              ← Tasks 1-4
├── requirements.txt               ← All dependencies
├── heart_disease_code/            ← Task 5 (CI/CD)
│   └── COMMANDS.md
├── api/                           ← Task 6 (Docker API)
│   └── COMMANDS.md
└── kubernetes/                    ← Task 7 (Minikube)
    └── COMMANDS.md
```

---

## 🚀 Quick Start (First Time Setup - REQUIRED!)

### 1. Create Virtual Environment & Install ALL Packages
```powershell
# Step 1: Create virtual environment
python -m venv venv

# Step 2: Activate it
.\venv\Scripts\Activate.ps1

# Step 3: Install ALL dependencies (includes pytest, flake8, jupyter, etc.)
pip install -r requirements.txt
```

**⚠️ IMPORTANT:** You MUST run `pip install -r requirements.txt` before running any commands!

This installs:
- pytest (for testing)
- flake8 (for linting)
- jupyter (for notebook)
- All ML libraries

---

## ✅ Task 1-4: Run Jupyter Notebook

### Run Notebook
```powershell
jupyter notebook assignment1.ipynb
```
**Execute all cells** (Shift + Enter on each cell or Cell → Run All)

### View MLflow Results
```powershell
mlflow ui
```
Open: http://localhost:5000

**Output Files:**
- `*.pkl` - Trained models
- `preprocessing_config.json` - Configuration

---

## ✅ Task 5: CI/CD & Testing

### Navigate to Code Folder
```powershell
cd heart_disease_code
```

### Run Unit Tests
```powershell
pytest -v
```
Expected: 12 passed

### Run Tests with Coverage
```powershell
pytest -v --cov=src --cov-report=html
```
View report: `htmlcov/index.html`

### Run Linting
```powershell
# Check for errors
flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics

# Full style check
flake8 src tests
```

### Go Back to Root
```powershell
cd ..
```

---

## 🐳 Task 6: Docker API

### Navigate to API Folder
```powershell
cd api
```

### Build Docker Image
```powershell
docker build -t heart-disease-api .
```
Time: ~3-5 minutes (first time)

### Run Container
```powershell
docker run -d -p 8000:8000 --name heart-api heart-disease-api
```

### Test API
```powershell
# Open in browser
http://localhost:8000/docs

# Or run test script
python test_api.py

# Or check health
curl http://localhost:8000/health
```

### View Metrics & Logs
```powershell
# View API metrics
http://localhost:8000/metrics

# View request logs
http://localhost:8000/logs
```

### Stop Container
```powershell
docker stop heart-api
docker rm heart-api
```

---

## ☸️ Task 7: Kubernetes Deployment (Minikube)

### Start Minikube
```powershell
minikube start
```

### Build Image in Minikube
```powershell
# Point Docker to Minikube's Docker
minikube docker-env | Invoke-Expression

# Build image
cd api
docker build -t heart-disease-api:v1 .
cd ..
```

### Deploy to Kubernetes
```powershell
# Apply deployment and service
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

### Verify Deployment
```powershell
# Check pods
kubectl get pods

# Check deployment
kubectl get deployments

# Check service
kubectl get services
```

### Access the API
```powershell
# Open service in browser
minikube service heart-disease-service

# Or get URL
minikube service heart-disease-service --url
```

### Test API
```powershell
# Get URL
$API_URL = minikube service heart-disease-service --url

# Test health
curl "$API_URL/health"

# View docs
# Open: $API_URL/docs in browser

# View metrics
# Open: $API_URL/metrics in browser

# View logs
# Open: $API_URL/logs in browser
```
, /health, /metrics, /logs
### Screenshots to Take
```powershell
kubectl get pods -o wide
kubectl get deployment heart-disease-deployment
kubectl get service heart-disease-service
# Also: Browser screenshots of /docs and /health endpoints
```

### Cleanup
```powershell
kubectl delete -f kubernetes/
minikube stop
```

---

## 📊 Assignment Tasks Checklist

- [ ] Task 1: Data Acquisition & EDA (5 marks)
  - Run: `jupyter notebook assignment1.ipynb` (Cells 1-28)

- [ ] Task 2: Models (8 marks)
  - Run: Notebook cells 29-42

- [ ] Task 3: MLflow (5 marks)
  - Run: Notebook cells 43-46
  - View: `mlflow ui`

- [ ] Task 4: Packaging (7 marks)
  - Run: Notebook cells 47-51
  - Files: `*.pkl`, `preprocessing_config.json`

- [ ] Task 5: CI/CD (8 marks)
  - Run: `cd heart_disease_code && pytest -v`
  - GitHub Actions: Auto-runs on push

- [ ] Task 6: Docker API
  - Run: `cd api && docker build -t heart-disease-api .`
  - Test: `docker run -p 8000:8000 heart-disease-api`

- [ ] Task 7: Kubernetes Deployment (7 marks)
  - Run: `minikube start`
  - Deploy: `kubectl apply -f kubernetes/`
  - Access: `minikube service heart-disease-service`

- [ ] Monitoring: Logging & Metrics
  - Metrics: `http://localhost:8000/metrics`
  - Logs: `http://localhost:8000/logs`
  - File: `api/api_logs.log`

---

## 🔍 Detailed Commands

See folder-specific COMMANDS.md:
- `heart_disease_code/COMMANDS.md` - Testing details
- `api/COMMANDS.md` - Docker details
- `kubernetes/COMMANDS.md` - Minikube deployment details

---

## 📦 Dependencies (in requirements.txt)

Core ML:
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- mlflow

Testing & Quality:
- pytest, pytest-cov
- flake8, pylint

Notebook:
- jupyter
