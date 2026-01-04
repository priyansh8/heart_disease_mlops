from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .data import load_dataset
from .features import build_preprocess
from .metrics import as_dict, evaluate_binary
from .models import get_model_specs


def cv_predict_proba(pipe: Pipeline, X, y, seed: int) -> np.ndarray:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")
    return proba[:, 1]


def train_and_log(data_path: str, outdir: Path, experiment: str, seed: int) -> Tuple[str, dict]:
    outdir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(data_path)
    preprocess = build_preprocess()

    mlflow.set_experiment(experiment)

    best_run_id = None
    best_metric = -1.0
    best_summary: dict = {}

    for spec in get_model_specs(seed):
        with mlflow.start_run(run_name=spec.name) as run:
            # Build full pipeline
            pipe = Pipeline(steps=[("preprocess", preprocess), ("model", spec.estimator)])

            # Log parameters (both our "documented" params and estimator params)
            mlflow.log_params({f"doc_{k}": v for k, v in spec.params.items()})
            mlflow.log_param("model_name", spec.name)

            # CV evaluation
            y_proba = cv_predict_proba(pipe, ds.X, ds.y, seed=seed)
            res = evaluate_binary(ds.y, y_proba)
            metrics = as_dict(res)
            mlflow.log_metrics(metrics)

            # Fit on full data
            pipe.fit(ds.X, ds.y)

            # Persist artifacts
            model_path = outdir / f"{spec.name}_pipeline.joblib"
            joblib.dump(pipe, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model_joblib")

            # Also log as MLflow model
            mlflow.sklearn.log_model(pipe, artifact_path="model_mlflow")

            # Track best by ROC-AUC
            if metrics["roc_auc"] > best_metric:
                best_metric = metrics["roc_auc"]
                best_run_id = run.info.run_id
                best_summary = {"run_id": best_run_id, "model_name": spec.name, **metrics}

    if best_run_id is None:
        raise RuntimeError("No successful MLflow runs created.")

    # Copy best pipeline to standard path for serving
    best_model_name = best_summary["model_name"]
    best_path = outdir / f"{best_model_name}_pipeline.joblib"
    serve_path = outdir / "model.joblib"
    joblib.dump(joblib.load(best_path), serve_path)

    # Save metrics summary
    (outdir / "metrics.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    return best_run_id, best_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/heart.csv")
    parser.add_argument("--outdir", type=Path, default=Path("artifacts"))
    parser.add_argument("--experiment", type=str, default="heart_disease")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Allow overriding tracking URI (default local ./mlruns)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    run_id, summary = train_and_log(args.data, args.outdir, args.experiment, args.seed)
    print(f"Best run: {run_id}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
