from __future__ import annotations

from pydantic import BaseModel, Field


class HeartFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: int = Field(..., description="0=female, 1=male")
    cp: int = Field(..., description="chest pain type (0-3)")
    trestbps: float = Field(..., ge=0)
    chol: float = Field(..., ge=0)
    fbs: int = Field(..., description="fasting blood sugar > 120 mg/dl (1=true; 0=false)")
    restecg: int = Field(..., description="resting ECG results (0-2)")
    thalach: float = Field(..., ge=0)
    exang: int = Field(..., description="exercise induced angina (1=yes; 0=no)")
    oldpeak: float = Field(..., ge=0)
    slope: int = Field(..., description="slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., description="number of major vessels (0-3) colored by flourosopy")
    thal: int = Field(..., description="thalassemia (typically 0-3 or encoded)")

    model_config = {"extra": "forbid"}


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
