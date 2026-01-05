# Docker API Commands

## Build Image
**From api/ folder:**
```powershell
cd api
docker build -t heart-disease-api .
```

**OR from root folder:**
```powershell
docker build -t heart-disease-api -f api/Dockerfile api/
```

## Run Container
```bash
docker run -d -p 8000:8000 --name heart-api heart-disease-api
```

## Test API
```bash
# Check health
curl http://localhost:8000/health

# Interactive docs
# Open: http://localhost:8000/docs

# Run tests
python test_api.py
```

## Sample Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```

## Stop/Clean
```bash
docker stop heart-api
docker rm heart-api
```
