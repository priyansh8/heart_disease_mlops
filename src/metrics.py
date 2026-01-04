from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    roc_auc: float


def evaluate_binary(y_true, y_proba) -> EvalResult:
    y_pred = (np.asarray(y_proba) >= 0.5).astype(int)
    return EvalResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
    )


def as_dict(res: EvalResult) -> Dict[str, float]:
    return {"accuracy": res.accuracy, "precision": res.precision, "recall": res.recall, "roc_auc": res.roc_auc}
