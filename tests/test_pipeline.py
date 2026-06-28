"""Unit tests for the diabetes prediction pipeline.

src/ is added to sys.path by the top-level conftest.py.
"""

import numpy as np

from config import CONFIG, resolve
from data_preprocessing import (
    build_preprocessor,
    clean_data,
    engineer_features,
    load_data,
    make_xy,
    split,
)

DATA = resolve(CONFIG["data"]["raw_path"])


def test_load_data_shape():
    df = load_data(DATA)
    assert df.shape == (768, 9)
    assert "Outcome" in df.columns


def test_clean_data_replaces_zeros():
    cleaned = clean_data(load_data(DATA))
    assert cleaned["Glucose"].isna().sum() > 0
    assert cleaned["Insulin"].isna().sum() > 0


def test_engineer_features_adds_columns():
    out = engineer_features(clean_data(load_data(DATA)))
    assert any(c.startswith("BMI_Category_") for c in out.columns)
    assert any(c.startswith("Glucose_Risk_") for c in out.columns)


def test_make_xy_separates_target():
    X, y = make_xy(DATA)
    assert "Outcome" not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y) == 768


def test_split_is_stratified():
    X, y = make_xy(DATA)
    X_tr, X_te, y_tr, y_te = split(X, y)
    assert len(X_tr) + len(X_te) == 768
    # Stratification keeps class balance roughly equal across splits.
    assert abs(y_tr.mean() - y_te.mean()) < 0.05


def test_preprocessor_imputes_and_scales():
    X, y = make_xy(DATA)
    feats = list(X.columns)
    pre = build_preprocessor(feats)
    Xt = pre.fit_transform(X)
    # No NaNs after imputation.
    assert not np.isnan(Xt).any()
    # Scaled columns are approximately mean 0.
    assert abs(Xt.mean()) < 1e-6


def test_config_has_required_keys():
    for key in ("seed", "data", "paths", "mlflow", "models", "risk_bands"):
        assert key in CONFIG


def test_risk_band_thresholds_ordered():
    assert CONFIG["risk_bands"]["low_max"] < CONFIG["risk_bands"]["moderate_max"]
