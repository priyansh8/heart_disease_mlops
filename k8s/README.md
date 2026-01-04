# Kubernetes notes

This setup uses `hostPath` to mount the trained model into the pod (good for local Minikube / Docker Desktop).

1) Copy your trained artifacts to `/tmp/heart-artifacts` on the machine running Kubernetes:
```bash
mkdir -p /tmp/heart-artifacts
cp artifacts/model.joblib /tmp/heart-artifacts/model.joblib
```

2) Apply manifests:
```bash
kubectl apply -f k8s/namespace.yaml
kubectl -n mlops-heart apply -f k8s/deployment.yaml
kubectl -n mlops-heart apply -f k8s/service.yaml
```

3) Port-forward:
```bash
kubectl -n mlops-heart port-forward svc/heart-api 8000:80
```

Then call `/predict` as usual.
