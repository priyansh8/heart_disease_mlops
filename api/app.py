"""
Cats vs Dogs Classification API
FastAPI service exposing a CNN for image predictions.
"""
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
import numpy as np
import os
import logging
import time
from datetime import datetime
from collections import defaultdict
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Metrics storage
metrics = {
    "total_requests": 0,
    "endpoint_counts": defaultdict(int),
    "status_counts": defaultdict(int),
    "total_response_time": 0.0,
    "request_log": []
}

# Initialize FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="API for predicting cats vs dogs from images using a simple CNN",
    version="1.0.0"
)

# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Update metrics
    metrics["total_requests"] += 1
    metrics["endpoint_counts"][request.url.path] += 1
    metrics["status_counts"][response.status_code] += 1
    metrics["total_response_time"] += process_time
    
    # Log request
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration": round(process_time, 4)
    }
    metrics["request_log"].append(log_entry)
    
    # Keep only last 100 logs in memory
    if len(metrics["request_log"]) > 100:
        metrics["request_log"] = metrics["request_log"][-100:]
    
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s")
    
    return response

# no Pydantic schema needed for image upload; we use UploadFile

# Load CNN model at startup
MODEL_DIR = "models"
cnn_model = None

try:
    print("Loading model...")
    cnn_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'cat_dog_model.h5'))
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    cnn_model = None

# Root endpoint
@app.get("/")
def read_root():
    """Welcome endpoint with API information"""
    return {
        "message": "Cats vs Dogs Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/predict": "Make prediction (POST, upload image)",
            "/docs": "Interactive API documentation",
            "/model/info": "Model information"
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    """Check if API and models are healthy"""
    models_loaded = cnn_model is not None
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "model_type": "Simple CNN" if models_loaded else None
    }

# Model info endpoint
@app.get("/model/info")
def model_info():
    """Get information about the loaded model"""
    if cnn_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Simple CNN",
        "input_shape": cnn_model.input_shape,
        "classes": ["cats", "dogs"],
        "description": "Binary image classifier for cats versus dogs"
    }

# Prediction endpoint expects an image file upload
@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """
    Make a prediction on an uploaded image. Returns the predicted label
    and confidence.
    """
    if cnn_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert('RGB')
        img = img.resize((224, 224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        features = np.expand_dims(arr, axis=0)

        probs = cnn_model.predict(features)[0][0]
        pred = int(probs >= 0.5)
        confidence = float(probs if pred == 1 else 1 - probs)

        return {
            "prediction": pred,
            "prediction_label": "dog" if pred == 1 else "cat",
            "confidence": round(confidence, 4),
            "probability": round(float(probs), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Metrics endpoint
@app.get("/metrics")
def get_metrics():
    """Get API usage metrics"""
    avg_response_time = (
        metrics["total_response_time"] / metrics["total_requests"]
        if metrics["total_requests"] > 0 else 0
    )
    
    return {
        "total_requests": metrics["total_requests"],
        "endpoints": dict(metrics["endpoint_counts"]),
        "status_codes": dict(metrics["status_counts"]),
        "average_response_time_seconds": round(avg_response_time, 4),
        "success_rate": f"{(metrics['status_counts'][200] / metrics['total_requests'] * 100) if metrics['total_requests'] > 0 else 0:.2f}%"
    }

# Logs endpoint
@app.get("/logs")
def get_logs(limit: int = 50):
    """Get recent API request logs"""
    return {
        "total_logs": len(metrics["request_log"]),
        "logs": metrics["request_log"][-limit:]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
