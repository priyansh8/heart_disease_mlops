"""Run EDA and save plots.

Produces:
- histograms for numeric features
- correlation heatmap (matplotlib)
- target class balance bar chart
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_histograms(df: pd.DataFrame, outdir: Path) -> None:
    numeric_cols = [c for c in df.columns if c != "target"]
    for col in numeric_cols:
        plt.figure()
        df[col].hist(bins=20)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(outdir / f"hist_{col}.png", dpi=180)
        plt.close()


def save_corr_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(outdir / "correlation_heatmap.png", dpi=180)
    plt.close()


def save_class_balance(df: pd.DataFrame, outdir: Path) -> None:
    counts = df["target"].value_counts().sort_index()
    plt.figure()
    counts.plot(kind="bar")
    plt.title("Class balance (target)")
    plt.xlabel("target")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outdir / "class_balance.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/heart.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    args.outdir.mkdir(parents=True, exist_ok=True)

    save_histograms(df, args.outdir)
    save_corr_heatmap(df, args.outdir)
    save_class_balance(df, args.outdir)

    print(f"Saved EDA figures to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
