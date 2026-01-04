"""Download Heart Disease UCI (processed Cleveland) dataset and write as CSV.

We convert the original multiclass label to binary:
- 0 => no disease
- 1..4 => disease present
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
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
    "target",
]


def download(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(UCI_URL, header=None, names=COLUMNS)
    df = df.replace("?", pd.NA)

    # Convert numeric columns safely
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/raw/heart.csv"))
    args = parser.parse_args()

    path = download(args.out)
    print(f"Wrote: {path} (rows={sum(1 for _ in open(path, 'r', encoding='utf-8'))-1})")


if __name__ == "__main__":
    main()
