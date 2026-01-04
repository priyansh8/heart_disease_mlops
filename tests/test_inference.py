import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.data import FEATURES
from src.inference import predict_one


def test_predict_one(tmp_path):
    # Simple model that outputs 0.5 proba always
    X = pd.DataFrame([{c: 0 for c in FEATURES}])
    y = [0]

    preprocess = ColumnTransformer([("num", Pipeline([("sc", StandardScaler(with_mean=False))]), FEATURES)])
    model = DummyClassifier(strategy="prior")
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X, y)

    p = tmp_path / "m.joblib"
    joblib.dump(pipe, p)

    pred = predict_one(pipe, {c: 0 for c in FEATURES})
    assert pred.label in (0, 1)
    assert 0.0 <= pred.confidence <= 1.0
