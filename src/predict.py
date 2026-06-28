"""
Inference for the Diabetes Risk Prediction project.

Loads the single self-contained pipeline (preprocess + model) saved by
train.py, so imputation and scaling are applied exactly as learned during
training. No hardcoded medians, no separate scaler to keep in sync.
"""

from __future__ import annotations

import argparse
import json
import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import CONFIG, resolve
from data_preprocessing import ZERO_AS_NAN, engineer_features

log = logging.getLogger(__name__)

PIPELINE_PATH = resolve(CONFIG["paths"]["models_dir"]) / "best_pipeline.pkl"
LOW_MAX = CONFIG["risk_bands"]["low_max"]
MOD_MAX = CONFIG["risk_bands"]["moderate_max"]


@lru_cache(maxsize=1)
def _load_pipeline(path: str = str(PIPELINE_PATH)):
    """Load and cache the fitted pipeline so repeated calls don't re-read disk."""
    return joblib.load(path)


def _band(p: float) -> str:
    if p < LOW_MAX:
        return "Low"
    if p < MOD_MAX:
        return "Moderate"
    return "High"


def predict(records: list[dict]) -> list[dict]:
    """Score a list of patient dicts (8 Pima features) -> risk results."""
    df = pd.DataFrame(records)
    df[ZERO_AS_NAN] = df[ZERO_AS_NAN].replace(0, np.nan)
    df = engineer_features(df)

    pipeline = _load_pipeline()

    # Align engineered columns to what the pipeline was trained on.
    expected = pipeline.named_steps["preprocess"].feature_names_in_
    for col in expected:
        if col not in df.columns:
            df[col] = 0
    df = df[list(expected)]

    proba = pipeline.predict_proba(df)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return [
        {"diabetes_risk": round(float(p), 4), "prediction": int(c), "risk_band": _band(p)}
        for p, c in zip(proba, preds)
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Run diabetes risk inference.")
    p.add_argument("--json", type=str, help="Path to a JSON file of patient records.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.json:
        records = json.loads(Path(args.json).read_text())
    else:
        records = [
            {
                "Pregnancies": 2,
                "Glucose": 148,
                "BloodPressure": 72,
                "SkinThickness": 35,
                "Insulin": 0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50,
            }
        ]
    print(json.dumps(predict(records), indent=2))


if __name__ == "__main__":
    main()
