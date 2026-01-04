from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from .data import FEATURES


@dataclass(frozen=True)
class Prediction:
    label: int
    confidence: float


def load_model(model_path: str):
    return joblib.load(model_path)


def predict_one(model, payload: Dict[str, float | int]) -> Prediction:
    # Validate schema
    missing = set(FEATURES) - set(payload.keys())
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")

    X = pd.DataFrame([{k: payload[k] for k in FEATURES}])
    proba = float(model.predict_proba(X)[:, 1][0])
    label = int(proba >= 0.5)
    confidence = float(proba if label == 1 else 1 - proba)
    return Prediction(label=label, confidence=confidence)
