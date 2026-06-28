"""
Data preprocessing for the Diabetes Risk Prediction project.

Key design choice: preprocessing (impute + scale) is packaged into a
scikit-learn ``Pipeline`` so that the EXACT same transformation learned at
training time is reused at inference time. This eliminates train/serve skew —
the previous version hardcoded imputation medians in the predict script, which
could silently drift from the training data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CONFIG, resolve

log = logging.getLogger(__name__)

COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

# Features where a value of 0 is clinically implausible -> treat as missing.
ZERO_AS_NAN = CONFIG["zero_as_nan"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and attach column names."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    log.info("Loaded data with shape %s", df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace clinically-impossible zeros with NaN."""
    df = df.copy()
    df[ZERO_AS_NAN] = df[ZERO_AS_NAN].replace(0, np.nan)
    log.info("Missing values after cleaning: %s", df.isna().sum().to_dict())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create clinically-meaningful derived categorical features (one-hot)."""
    df = df.copy()
    df["BMI_Category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 30, 45, 60, 120],
        labels=["Young", "Adult", "Middle", "Senior"],
    )
    df["Glucose_Risk"] = pd.cut(
        df["Glucose"],
        bins=[0, 99, 125, 200, 1000],
        labels=["Normal", "Prediabetic", "Diabetic", "Severe"],
    )
    df = pd.get_dummies(df, columns=["BMI_Category", "Age_Group", "Glucose_Risk"], drop_first=True)
    return df


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    """Return a ColumnTransformer that imputes (median) then scales all features.

    Wrapping this in a fitted transformer means inference reuses the learned
    medians + scaling automatically — no hardcoded constants.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipe, feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_xy(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load -> clean -> engineer features, returning X (raw, unscaled) and y."""
    df = engineer_features(clean_data(load_data(path)))
    y = df["Outcome"].astype(int)
    X = df.drop(columns=["Outcome"])
    return X, y


def split(X: pd.DataFrame, y: pd.Series, test_size: float | None = None, seed: int | None = None):
    """Stratified train/test split using config defaults."""
    test_size = test_size if test_size is not None else CONFIG["data"]["test_size"]
    seed = seed if seed is not None else CONFIG["seed"]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    X, y = make_xy(resolve(CONFIG["data"]["raw_path"]))
    X_tr, X_te, y_tr, y_te = split(X, y)
    log.info("X_train %s | X_test %s", X_tr.shape, X_te.shape)
