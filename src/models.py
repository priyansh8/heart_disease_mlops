from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    params: Dict[str, object]


def get_model_specs(seed: int) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="logreg",
            estimator=LogisticRegression(max_iter=2000, n_jobs=None),
            params={"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
        ),
        ModelSpec(
            name="rf",
            estimator=RandomForestClassifier(
                n_estimators=300,
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced",
            ),
            params={"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        ),
    ]
