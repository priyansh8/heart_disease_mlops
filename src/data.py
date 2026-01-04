from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET = "target"


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.Series


def load_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)

    # Ensure expected schema
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")

    # Basic NA handling (median for numeric)
    df = df.copy()
    for col in FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[FEATURES]
    y = df[TARGET].astype(int)
    return Dataset(X=X, y=y)


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET].astype(int)
