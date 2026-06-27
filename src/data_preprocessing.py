"""
Data preprocessing module for the Diabetes Readmission Prediction project.

Loads the Pima Indians Diabetes dataset, treats clinically-impossible zeros
as missing values, imputes them, scales features, and returns train/test splits.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
ZERO_AS_NAN = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and attach column names."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    log.info("Loaded data with shape %s", df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace clinically-impossible zeros with NaN."""
    df = df.copy()
    df[ZERO_AS_NAN] = df[ZERO_AS_NAN].replace(0, np.nan)
    log.info("Missing values after cleaning:\n%s", df.isna().sum().to_dict())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create clinically-meaningful derived features."""
    df = df.copy()
    df["BMI_Category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )
    df["Age_Group"] = pd.cut(
        df["Age"], bins=[0, 30, 45, 60, 120], labels=["Young", "Adult", "Middle", "Senior"]
    )
    df["Glucose_Risk"] = pd.cut(
        df["Glucose"],
        bins=[0, 99, 125, 200, 1000],
        labels=["Normal", "Prediabetic", "Diabetic", "Severe"],
    )
    df = pd.get_dummies(df, columns=["BMI_Category", "Age_Group", "Glucose_Risk"], drop_first=True)
    return df


def prepare(path: str | Path, test_size: float = 0.2, random_state: int = 42):
    """Full preprocessing pipeline -> X_train, X_test, y_train, y_test, scaler."""
    df = engineer_features(clean_data(load_data(path)))

    y = df["Outcome"].astype(int)
    X = df.drop(columns=["Outcome"])

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    log.info("Train shape %s, Test shape %s", X_train_s.shape, X_test_s.shape)
    return X_train_s, X_test_s, y_train, y_test, scaler


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parent.parent / "data" / "diabetes_raw.csv")
