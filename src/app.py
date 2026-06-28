"""
FastAPI serving layer for the Diabetes Risk Prediction model.

Exposes a /predict endpoint that wraps the same predict() used in batch mode,
so online and offline scoring share one code path. This is the artifact that
maps onto AWS (Lambda/ECS/SageMaker endpoint) in the cloud notes.

Run locally:  uvicorn app:app --reload --app-dir src
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from predict import predict

app = FastAPI(title="Diabetes Risk Prediction API", version="2.0.0")


class Patient(BaseModel):
    Pregnancies: int = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: int = Field(..., ge=0)


class PredictResponse(BaseModel):
    diabetes_risk: float
    prediction: int
    risk_band: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=list[PredictResponse])
def predict_endpoint(patients: list[Patient]) -> list[dict]:
    return predict([p.model_dump() for p in patients])
