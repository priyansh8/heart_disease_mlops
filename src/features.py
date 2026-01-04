from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FEATURES

# In this dataset, all features are numeric/categorical encoded as ints already.
# We'll still treat integer-coded categoricals as categorical for one-hot encoding.
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]


def build_preprocess() -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipe = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL),
        ],
        remainder="drop",
    )
    return preprocess
