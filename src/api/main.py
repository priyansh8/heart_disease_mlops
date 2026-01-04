from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, Request, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .metrics import REQUEST_COUNT, REQUEST_LATENCY, PREDICTIONS, now
from .schemas import HeartFeatures, PredictResponse
from ..inference import load_model, predict_one

logger = logging.getLogger("heart_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")

app = FastAPI(title="Heart Disease Risk API", version="1.0.0")

_model = None


@app.on_event("startup")
def _startup() -> None:
    global _model
    _model = load_model(MODEL_PATH)
    logger.info("Loaded model from %s", MODEL_PATH)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    endpoint = request.url.path
    start = now()
    try:
        response: Response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        dur = now() - start
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(dur)
        REQUEST_COUNT.labels(endpoint=endpoint, method=request.method, status=status if 'status' in locals() else "500").inc()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: HeartFeatures) -> PredictResponse:
    pred = predict_one(_model, payload.model_dump())
    PREDICTIONS.labels(prediction=str(pred.label)).inc()
    logger.info("Prediction=%s confidence=%.4f payload=%s", pred.label, pred.confidence, payload.model_dump())
    return PredictResponse(prediction=pred.label, confidence=pred.confidence)


@app.get("/metrics")
def metrics() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
