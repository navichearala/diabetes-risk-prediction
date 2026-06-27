"""Inference script for the Diabetes Readmission Prediction project."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_preprocessing import ZERO_AS_NAN, engineer_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"


def predict(records: list[dict]) -> list[dict]:
    """records: list of dicts with the 8 Pima features."""
    df = pd.DataFrame(records)
    df[ZERO_AS_NAN] = df[ZERO_AS_NAN].replace(0, np.nan)
    df = engineer_features(df)
    # Use sensible clinical medians (from training set) for imputation at inference time
    medians = {"Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 29.0,
               "Insulin": 125.0, "BMI": 32.0}
    for col, val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    df = df.fillna(0)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Align columns to scaler training order
    expected = scaler.feature_names_in_
    for col in expected:
        if col not in df.columns:
            df[col] = 0
    df = df[expected]

    X_scaled = scaler.transform(df)
    proba = model.predict_proba(X_scaled)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return [
        {"diabetes_risk": float(p), "prediction": int(c), "risk_band": _band(p)}
        for p, c in zip(proba, preds)
    ]


def _band(p: float) -> str:
    if p < 0.3:
        return "Low"
    if p < 0.6:
        return "Moderate"
    return "High"


if __name__ == "__main__":
    sample = [{
        "Pregnancies": 2, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
        "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50,
    }]
    print(json.dumps(predict(sample), indent=2))
