"""
Heart Disease Prediction API
FastAPI application for serving ML model predictions
"""
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List
import os
import logging
import time
from datetime import datetime
from collections import defaultdict

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
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using Random Forest model",
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

# Define input data schema
class PatientData(BaseModel):
    age: float = Field(..., description="Age in years", ge=0, le=120)
    sex: float = Field(..., description="Sex (1 = male; 0 = female)", ge=0, le=1)
    cp: float = Field(..., description="Chest pain type (0-3)", ge=0, le=3)
    trestbps: float = Field(..., description="Resting blood pressure (mm Hg)", ge=0)
    chol: float = Field(..., description="Serum cholesterol (mg/dl)", ge=0)
    fbs: float = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)", ge=0, le=1)
    restecg: float = Field(..., description="Resting ECG results (0-2)", ge=0, le=2)
    thalach: float = Field(..., description="Maximum heart rate achieved", ge=0, le=250)
    exang: float = Field(..., description="Exercise induced angina (1 = yes; 0 = no)", ge=0, le=1)
    oldpeak: float = Field(..., description="ST depression induced by exercise", ge=0)
    slope: float = Field(..., description="Slope of peak exercise ST segment (0-2)", ge=0, le=2)
    ca: float = Field(..., description="Number of major vessels colored by fluoroscopy (0-3)", ge=0, le=3)
    thal: float = Field(..., description="Thalassemia (0-3)", ge=0, le=3)
    
    class Config:
        schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        }

# Load models at startup
MODEL_DIR = "models"

try:
    print("Loading models...")
    with open(f'{MODEL_DIR}/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    # Note: imputer not needed for API - it only handles ca/thal columns from training data
    # API input validation ensures no missing values
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    rf_model = None

# Root endpoint
@app.get("/")
def read_root():
    """Welcome endpoint with API information"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/predict": "Make prediction (POST)",
            "/docs": "Interactive API documentation",
            "/model/info": "Model information"
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    """Check if API and models are healthy"""
    models_loaded = rf_model is not None
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "model_type": "Random Forest" if models_loaded else None
    }

# Model info endpoint
@app.get("/model/info")
def model_info():
    """Get information about the loaded model"""
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Random Forest Classifier",
        "n_estimators": rf_model.n_estimators,
        "features": 13,
        "target_classes": ["No Disease", "Disease"],
        "training_accuracy": "88.5%",
        "description": "Predicts presence of heart disease based on 13 clinical features"
    }

# Prediction endpoint
@app.post("/predict")
def predict(patient: PatientData) -> Dict:
    """
    Make heart disease prediction for a patient
    
    Returns:
        - prediction: 0 (No Disease) or 1 (Disease)
        - prediction_label: Human-readable prediction
        - confidence: Confidence score (0-1)
        - probabilities: Probability for each class
    """
    # Check if models are loaded
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert input to array
        features = np.array([[
            patient.age, patient.sex, patient.cp, patient.trestbps,
            patient.chol, patient.fbs, patient.restecg, patient.thalach,
            patient.exang, patient.oldpeak, patient.slope, patient.ca,
            patient.thal
        ]])
        
        # Make prediction directly (no imputation needed - API validates complete input)
        prediction = rf_model.predict(features)[0]
        probabilities = rf_model.predict_proba(features)[0]
        
        # Get confidence (probability of predicted class)
        confidence = float(probabilities[prediction])
        
        # Format response
        response = {
            "prediction": int(prediction),
            "prediction_label": "Disease" if prediction == 1 else "No Disease",
            "confidence": round(confidence, 4),
            "probabilities": {
                "no_disease": round(float(probabilities[0]), 4),
                "disease": round(float(probabilities[1]), 4)
            },
            "model_used": "Random Forest",
            "input_features": {
                "age": patient.age,
                "sex": "Male" if patient.sex == 1 else "Female",
                "chest_pain_type": int(patient.cp),
                "resting_bp": patient.trestbps,
                "cholesterol": patient.chol
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint (bonus)
@app.post("/predict/batch")
def predict_batch(patients: list[PatientData]) -> Dict:
    """
    Make predictions for multiple patients at once
    
    Returns list of predictions
    """
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    results = []
    for patient in patients:
        try:
            # Reuse single prediction logic
            prediction_result = predict(patient)
            results.append(prediction_result)
        except Exception as e:
            results.append({"error": str(e)})
    
    return {
        "total_patients": len(patients),
        "predictions": results
    }

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
