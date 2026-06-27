"""
Training pipeline with MLflow tracking and model registry (MLOps).

Trains Logistic Regression, Random Forest, and Gradient Boosting with
hyperparameter tuning, logs metrics/params/artifacts to MLflow, and saves
the best model.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data_preprocessing import prepare

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
MODELS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_DB = ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("diabetes_readmission_prediction")


MODEL_GRID = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, solver="liblinear"),
        {"C": [0.1, 1, 10], "penalty": ["l1", "l2"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [200, 400], "max_depth": [None, 8, 14], "min_samples_split": [2, 5]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [150, 300], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
    ),
}


def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def plot_confusion(y_true, y_pred, name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No DM", "DM"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    path = FIG_DIR / f"cm_{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc_curves(rocs: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc) in rocs.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Diabetes Prediction Models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_feature_importance(model, feature_names, name: str):
    if not hasattr(model, "feature_importances_"):
        return None
    fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    fi.plot(kind="barh", ax=ax, color="#2E86AB")
    ax.set_title(f"Top Feature Importances — {name}")
    fig.tight_layout()
    path = FIG_DIR / f"feature_importance_{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def train_all(data_path: Path):
    X_train, X_test, y_train, y_test, scaler = prepare(data_path)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results, rocs, best = {}, {}, {"score": -1, "name": None, "model": None}

    for name, (estimator, grid) in MODEL_GRID.items():
        with mlflow.start_run(run_name=name):
            log.info("Training %s ...", name)
            gs = GridSearchCV(estimator, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
            gs.fit(X_train, y_train)

            model = gs.best_estimator_
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = evaluate(y_test, y_pred, y_proba)
            results[name] = metrics

            mlflow.log_params(gs.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name="model")

            cm_path = plot_confusion(y_test, y_pred, name)
            mlflow.log_artifact(str(cm_path))

            fi_path = plot_feature_importance(model, X_train.columns, name)
            if fi_path:
                mlflow.log_artifact(str(fi_path))

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            rocs[name] = (fpr, tpr, metrics["roc_auc"])

            log.info("%s -> %s", name, metrics)

            if metrics["roc_auc"] > best["score"]:
                best.update(score=metrics["roc_auc"], name=name, model=model,
                            params=gs.best_params_, metrics=metrics)

    roc_path = FIG_DIR / "05_roc_curves.png"
    plot_roc_curves(rocs, roc_path)

    # Persist artifacts
    joblib.dump(best["model"], MODELS_DIR / "best_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    summary = {
        "best_model": best["name"],
        "best_params": best["params"],
        "best_metrics": best["metrics"],
        "all_results": results,
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== TRAINING SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))

    # Register the best model
    with mlflow.start_run(run_name=f"best_{best['name']}_registry"):
        mlflow.log_params(best["params"])
        mlflow.log_metrics(best["metrics"])
        mlflow.sklearn.log_model(
            best["model"], name="model", registered_model_name="DiabetesReadmissionBest"
        )

    return summary


if __name__ == "__main__":
    train_all(ROOT / "data" / "diabetes_raw.csv")
