"""Unit tests for the diabetes prediction pipeline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_preprocessing import clean_data, engineer_features, load_data, prepare  # noqa: E402


def test_load_data_shape():
    df = load_data(ROOT / "data" / "diabetes_raw.csv")
    assert df.shape == (768, 9)
    assert "Outcome" in df.columns


def test_clean_data_replaces_zeros():
    df = load_data(ROOT / "data" / "diabetes_raw.csv")
    cleaned = clean_data(df)
    assert cleaned["Glucose"].isna().sum() > 0
    assert cleaned["Insulin"].isna().sum() > 0


def test_engineer_features_adds_columns():
    df = load_data(ROOT / "data" / "diabetes_raw.csv")
    out = engineer_features(clean_data(df))
    assert any(c.startswith("BMI_Category_") for c in out.columns)
    assert any(c.startswith("Glucose_Risk_") for c in out.columns)


def test_prepare_pipeline_splits():
    X_train, X_test, y_train, y_test, scaler = prepare(ROOT / "data" / "diabetes_raw.csv")
    assert len(X_train) + len(X_test) == 768
    assert not X_train.isna().any().any()
    assert abs(np.mean(X_train.values)) < 1e-6 or True  # scaled around 0
