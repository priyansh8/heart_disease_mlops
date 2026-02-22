# MLOps Assignment – Cats vs Dogs Image Classification

**M1: Model Development & Experiment Tracking**

---

## Project Structure

## Project Structure
```
Image_classification_mlops/
├── PetImages/                    ← Raw dataset (DVC-tracked)
│   ├── Cat/                      ← ~12,500 cat images
│   └── Dog/                      ← ~12,500 dog images
├── cats_dogs_dataset/            ← Preprocessed split (cats/ dogs/ subfolders)
│   ├── cats/
│   └── dogs/
├── cats_dogs_code/               ← Source package + unit tests
│   ├── src/
│   │   ├── preprocessing.py      ← Image loading, 224x224 resize, augmentation, split
│   │   └── training.py           ← CNN model build/train/evaluate/save
│   ├── tests/
│   │   ├── test_preprocessing.py ← 9 unit tests (augmentation, loading, splitting)
│   │   └── test_model.py         ← 2 unit tests (build, train, evaluate, save/load)
│   ├── pytest.ini
│   └── .flake8
├── .github/workflows/
│   ├── ci.yml                    ← CI: lint → test → docker build
│   └── ml_pipeline.yml           ← Full ML pipeline: lint → test → train + MLflow
├── train_pipeline.py             ← End-to-end training with MLflow tracking
├── cat_dog_model.h5              ← Trained Keras CNN model
├── PetImages.dvc                 ← DVC tracking file for the dataset
├── requirements.txt
└── README.md
```

## For final report and Screenshots please refer the 'screenshorts' folder

---

## M1 Completion Status

| Task | Status | Details |
|------|--------|---------|
| **Git versioning** | ✅ Done | Full git history, `.gitignore` configured |
| **DVC dataset versioning** | ✅ Done | `PetImages.dvc` tracks 24,998 images (848 MB) |
| **Image preprocessing** | ✅ Done | 224×224 RGB resize, 80/10/10 split, 5 augmentations |
| **Baseline CNN model** | ✅ Done | Simple CNN, trained, saved as `cat_dog_model.h5` |
| **MLflow experiment tracking** | ✅ Done | Logs hyperparams, per-epoch metrics, test metrics, confusion matrix, training curves, Keras model artifact |
| **Unit tests** | ✅ Done | 11/11 tests passing |
| **CI/CD pipeline** | ✅ Done | GitHub Actions: lint → test → train |

---

## Quick Start

### 1. Setup Virtual Environment and Install Dependencies

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

The raw dataset is tracked with DVC. Populate `cats_dogs_dataset/` with:
- `cats_dogs_dataset/cats/` → copy images from `PetImages/Cat/`
- `cats_dogs_dataset/dogs/` → copy images from `PetImages/Dog/`

Or use DVC (after configuring a remote):
```powershell
pip install dvc
dvc pull  # pulls cats_dogs_dataset from remote
```

### 3. Train Model with MLflow Tracking

```powershell
python train_pipeline.py
```

**What this does:**
- Loads images from `cats_dogs_dataset/` and resizes to 224×224 RGB
- Splits 80% train / 10% val / 10% test (stratified)
- Applies 5 data augmentation techniques to training set (2× multiplier)
- Builds and trains a simple CNN (Adam optimizer, binary crossentropy)
- Logs to MLflow: all hyperparameters, per-epoch train/val loss & accuracy, test metrics (accuracy, precision, recall, F1, ROC-AUC)
- Saves confusion matrix and training curves as artifacts
- Saves `cat_dog_model.h5` (Keras HDF5 format)

**Output artifacts:**
- `cat_dog_model.h5` — trained model
- `training_config.json` — run configuration and all metrics
- `confusion_matrix.png` — test set confusion matrix
- `training_curves.png` — loss and accuracy curves
- `mlruns/` — MLflow experiment logs

### 4. View MLflow Results

```powershell
mlflow ui
```

Open: [http://localhost:5000](http://localhost:5000)

You will see the experiment `cats_vs_dogs_classification` with:
- All hyperparameters logged
- Per-epoch training and validation loss/accuracy curves
- Final test metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix image
- Training curves image
- Saved Keras model artifact

---

## Unit Tests

### Run All Tests
```powershell
cd cats_dogs_code
python -m pytest tests/ -v
```

### Run with Coverage
```powershell
python -m pytest tests/ -v --cov=src --cov-report=html
```
View coverage report at `cats_dogs_code/htmlcov/index.html`.

### Run Linting
```powershell
flake8 src tests
```

**Test suite (11 tests):**
- `test_model.py` — CNN build/train, evaluate, save/load
- `test_preprocessing.py` — image loading, split, preprocess, augment (shape, range, reproducibility, size, labels, shuffling, metadata)

---

## DVC Dataset Versioning

The dataset is tracked with DVC. `PetImages.dvc` records the MD5 hash, size (848 MB), and file count (24,998 images).

```powershell
# Check DVC status
dvc status

# Add/update dataset tracking
dvc add PetImages

# Configure a remote storage (e.g., S3, GDrive, local)
dvc remote add -d myremote s3://your-bucket/dvc-store
dvc push           # push data to remote
dvc pull           # pull data from remote
```

---

## Data Augmentation

Applied to training set with 2× multiplier:

| Technique | Details |
|-----------|---------|
| Horizontal flip | 50% probability |
| Vertical flip | 50% probability |
| Rotation | ±15°, 50% probability |
| Brightness | 0.8–1.2× factor, 50% probability |
| Contrast | 0.8–1.2× factor, 50% probability |

---

## Model Architecture (Simple CNN)

```
Input: (224, 224, 3)
Conv2D(32, 3×3, relu) → MaxPooling(2×2)
Conv2D(64, 3×3, relu) → MaxPooling(2×2)
Flatten → Dense(64, relu) → Dense(1, sigmoid)

Loss: binary_crossentropy | Optimizer: adam
```

---

## CI/CD Pipeline (GitHub Actions)

**`.github/workflows/ml_pipeline.yml`** (full pipeline):
1. **Lint** — flake8 syntax + style, pylint
2. **Test** — pytest with coverage report (uploaded as artifact)
3. **Train** — runs `train_pipeline.py` with MLflow, uploads `cat_dog_model.h5`, plots, config, and `mlruns/`

**`.github/workflows/ci.yml`** (fast CI):
- lint → test → docker build

---

## Dependencies (`requirements.txt`)

| Package | Purpose |
|---------|---------|
| tensorflow | CNN model (Keras API) |
| pillow, opencv-python | Image processing |
| scikit-learn | Train/val/test splits, metrics |
| mlflow | Experiment tracking |
| numpy, matplotlib | Numerics and plots |
| dvc | Dataset versioning |
| pytest, pytest-cov, flake8 | Testing and linting |
| fastapi, uvicorn | Inference API server |
| requests | API test client |

---

## M2: Model Packaging & Containerization

### M2 Completion Status

| Task | Status | Details |
|------|--------|---------|
| **FastAPI inference service** | ✅ Done | `api/app.py` — `/health` + `/predict` + `/model/info` endpoints |
| **Environment specification** | ✅ Done | `api/requirements.txt` — all ML libs pinned with versions |
| **Dockerfile** | ✅ Done | `api/Dockerfile` — reproducible container with health check |
| **Local build & verify** | ✅ Done | Build + run commands below |

---

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and endpoint listing |
| GET | `/health` | Health check — confirms model is loaded |
| POST | `/predict` | Upload image → returns label + confidence + probability |
| GET | `/model/info` | Model type, input shape, class names |
| GET | `/metrics` | Request counts, response times, status codes |
| GET | `/logs` | Recent request logs |

### Files

```
api/
├── app.py              ← FastAPI service (image preprocessing + CNN inference)
├── Dockerfile          ← Container definition (python:3.9-slim base)
├── requirements.txt    ← Pinned versions: fastapi, uvicorn, tensorflow, pillow, numpy
├── .dockerignore       ← Excludes venv, __pycache__, test files
├── test_api.py         ← Test script for /health, /model/info, /predict
└── models/
    └── cat_dog_model.h5  ← Trained CNN model (copied from training)
```

---

### Build and Run Docker Container

```powershell
# Navigate to api/ folder
cd api

# Build the Docker image
docker build -t cats-dogs-api:latest .

# Run the container
docker run -d -p 8000:8000 --name cats-api cats-dogs-api:latest

# Check container is running
docker ps
```

---

### Verify Predictions

```powershell
# Health check
curl http://localhost:8000/health
# Response: {"status":"healthy","models_loaded":true,"model_type":"Simple CNN"}

# Model info
curl http://localhost:8000/model/info

# Predict from an image file (replace path with a real image)
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/cat.jpg"
# Response: {"prediction":0,"prediction_label":"cat","confidence":0.92,"probability":0.08}

# Or open the interactive docs in a browser
# http://localhost:8000/docs
```

### Run the Test Script (after starting the container)

```powershell
python test_api.py
# Runs: test_health, test_model_info, test_prediction
```

---

### Stop and Remove Container

```powershell
docker stop cats-api
docker rm cats-api
```

---

### Run API Locally Without Docker

```powershell
cd api
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# Open: http://localhost:8000/docs
```

---

### Environment Specification (api/requirements.txt)

All versions pinned for reproducibility:

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
tensorflow==2.14.0
pillow==10.1.0
numpy==1.26.2
requests==2.31.0
```

---

## M3: CI Pipeline for Build, Test & Image Creation

### M3 Completion Status

| Task | Status | Details |
|------|--------|---------|
| **Automated unit tests** | ✅ Done | 11 tests — preprocessing (9) + model (2) — run via pytest |
| **CI pipeline (GitHub Actions)** | ✅ Done | `.github/workflows/ci.yml` — 3 jobs triggered on push/PR |
| **Artifact publishing (GHCR)** | ✅ Done | Pushes to `ghcr.io/<owner>/cats-dogs-api` on merge to main |

---

### CI Pipeline — `.github/workflows/ci.yml`

3 sequential jobs, triggered on every push and pull request:

```
push / PR
    │
    ▼
┌─────────────────────────────┐
│  Job 1: test                │  lint (flake8) + pytest (11 tests) + coverage artifact
└──────────────┬──────────────┘
               │ needs: test
               ▼
┌─────────────────────────────┐
│  Job 2: build               │  docker build (validation, no push)
└──────────────┬──────────────┘
               │ needs: build + push to main only
               ▼
┌─────────────────────────────┐
│  Job 3: publish             │  docker push → ghcr.io/<owner>/cats-dogs-api
└─────────────────────────────┘
```

### Job Details

| Job | What it does |
|-----|-------------|
| `test` | Checkout → install deps → flake8 lint → `pytest tests/ -v --cov=src` → upload `coverage.xml` artifact |
| `build` | Checkout → Docker Buildx → build image (validates Dockerfile, uses GHA cache) |
| `publish` | Login to GHCR with `GITHUB_TOKEN` → build + push with `sha-` and `latest` tags |

### No Extra Secrets Needed

The publish job uses `secrets.GITHUB_TOKEN` (automatically available in every GitHub Actions run) — **no manual secret setup required** for GHCR.

---

### Triggering the Pipeline

**You need to do (manual steps):**

1. Push to GitHub:
```powershell
git push origin feature/assignment2-cats
```

2. Go to **GitHub → your repo → Actions tab** — watch the 3 jobs run

3. After merge to main, the image appears at:
```
ghcr.io/<your-github-username>/cats-dogs-api:latest
```

4. Pull it anywhere:
```bash
docker pull ghcr.io/<your-github-username>/cats-dogs-api:latest
```

---

### Published Image Tags

| Tag | When created |
|-----|-------------|
| `latest` | Every push to main |
| `sha-<commit-hash>` | Every push to main (pinned version) |

---

### Unit Tests (M3 Task 1)

```powershell
cd cats_dogs_code
python -m pytest tests/ -v
```

**Coverage:**
- `test_preprocessing.py` — 9 tests covering `load_images_from_folder`, `split_dataset`, `preprocess_single_image`, `augment_image`, `augment_dataset`, `get_augmentation_metadata`
- `test_model.py` — 2 tests covering `build_simple_cnn`, `train_cnn`, `evaluate_model`, `save_model`, `load_model`
