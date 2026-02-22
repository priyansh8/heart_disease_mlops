# Kubernetes Deployment Commands

## Prerequisites
```powershell
# Verify Minikube is installed
minikube version

# Verify kubectl is installed
kubectl version --client
```

## Step 1: Start Minikube
```powershell
minikube start
```

## Step 2: Build Docker Image in Minikube
```powershell
# Point Docker to Minikube's Docker daemon
minikube docker-env | Invoke-Expression

# Build image inside Minikube
cd api
docker build -t cats-dogs-api:v1 .
cd ..
```

## Step 3: Deploy to Kubernetes
```powershell
# Apply deployment
kubectl apply -f kubernetes/deployment.yaml

# Apply service
kubectl apply -f kubernetes/service.yaml
```

## Step 4: Verify Deployment
```powershell
# Check pods are running
kubectl get pods

# Check deployment
kubectl get deployments

# Check service
kubectl get services
```

## Step 5: Access the API
```powershell
# Get the service URL (opens in browser)
minikube service heart-disease-service

# OR get the URL without opening browser
minikube service heart-disease-service --url
```

## Test the API
```powershell
# Get the URL
$API_URL = minikube service heart-disease-service --url

# Test health endpoint
curl "$API_URL/health"

# Test prediction (upload local image)
curl -X POST "$API_URL/predict" -F "file=@path/to/image.jpg"
```

## Screenshots to Take
```powershell
# 1. Running pods
kubectl get pods -o wide

# 2. Deployment status
kubectl get deployment heart-disease-deployment

# 3. Service details
kubectl get service heart-disease-service

# 4. Pod logs
kubectl logs -l app=heart-disease-api

# 5. API docs (open in browser)
minikube service heart-disease-service
# Navigate to /docs endpoint
```

## Cleanup
```powershell
# Delete resources
kubectl delete -f kubernetes/service.yaml
kubectl delete -f kubernetes/deployment.yaml

# Stop Minikube
minikube stop

# Delete Minikube cluster
minikube delete
```
