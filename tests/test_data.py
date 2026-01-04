import pandas as pd
import pytest

from src.data import FEATURES, TARGET, load_dataset


def test_load_dataset_schema(tmp_path):
    df = pd.DataFrame([{**{c: 1 for c in FEATURES}, TARGET: 0}])
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)

    ds = load_dataset(str(p))
    assert list(ds.X.columns) == FEATURES
    assert ds.y.iloc[0] in (0, 1)


def test_load_dataset_missing_cols(tmp_path):
    df = pd.DataFrame([{"age": 1, TARGET: 0}])
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_dataset(str(p))
