# Docker API Commands

## Build Image
**From api/ folder:**
```powershell
cd api
docker build -t cats-dogs-api .
```

**OR from root folder:**
```powershell
docker build -t cats-dogs-api -f api/Dockerfile api/
```

## Run Container
```bash
docker run -d -p 8000:8000 --name cats-api cats-dogs-api
```

## Test API
```bash
# Check health
curl http://localhost:8000/health

# Interactive docs
# Open: http://localhost:8000/docs

# Run tests (makes an image upload)
python test_api.py
```

## Sample Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.jpg"
```
## Stop/Clean
```bash
docker stop cats-api
docker rm cats-api
```
