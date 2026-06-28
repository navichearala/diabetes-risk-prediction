"""
Training pipeline with MLflow tracking and model registry (MLOps).

Improvements over v1:
  * Every model is a full sklearn Pipeline (preprocess + estimator), so the
    saved artifact is self-contained — no separate scaler.pkl to keep in sync.
  * All hyperparameters, paths, seeds, and CV settings come from config.yaml.
  * CLI flags (--data, --config, --no-mlflow) for flexible runs.
  * GridSearchCV runs over the *pipeline*, preventing data leakage from the
    imputer/scaler being fit on the full set before CV.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from config import CONFIG, SEED, resolve
from data_preprocessing import build_preprocessor, make_xy, split

log = logging.getLogger(__name__)

MODELS_DIR = resolve(CONFIG["paths"]["models_dir"])
FIG_DIR = resolve(CONFIG["paths"]["figures_dir"])


def build_estimators() -> dict:
    """Instantiate base estimators; grids are read from config (prefixed for pipeline)."""
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, solver="liblinear"),
        "random_forest": RandomForestClassifier(random_state=SEED),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }


def prefixed_grid(name: str) -> dict:
    """Prefix config grid keys with 'model__' so GridSearchCV targets the pipeline step."""
    return {f"model__{k}": v for k, v in CONFIG["models"][name].items()}


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def plot_confusion(y_true, y_pred, name: str) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No DM", "DM"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    path = FIG_DIR / f"cm_{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc_curves(rocs: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc) in rocs.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Diabetes Prediction Models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_feature_importance(pipeline: Pipeline, feature_names, name: str):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    fi = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=True)
        .tail(15)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    fi.plot(kind="barh", ax=ax, color="#2E86AB")
    ax.set_title(f"Top Feature Importances - {name}")
    fig.tight_layout()
    path = FIG_DIR / f"feature_importance_{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def train_all(data_path: Path, use_mlflow: bool = True) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X, y = make_xy(data_path)
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = split(X, y)

    cv = StratifiedKFold(n_splits=CONFIG["cv"]["n_splits"], shuffle=True, random_state=SEED)
    scoring = CONFIG["cv"]["scoring"]

    if use_mlflow:
        mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])
        mlflow.set_experiment(CONFIG["mlflow"]["experiment_name"])

    results, rocs, best = {}, {}, {"score": -1.0, "name": None, "pipeline": None}

    for name, estimator in build_estimators().items():
        log.info("Training %s ...", name)
        pipe = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(feature_names)),
                ("model", estimator),
            ]
        )
        gs = GridSearchCV(pipe, prefixed_grid(name), cv=cv, scoring=scoring, n_jobs=-1)

        run_ctx = mlflow.start_run(run_name=name) if use_mlflow else _NullCtx()
        with run_ctx:
            gs.fit(X_train, y_train)
            pipeline = gs.best_estimator_
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            metrics = evaluate(y_test, y_pred, y_proba)
            results[name] = metrics
            log.info("%s -> %s", name, metrics)

            if use_mlflow:
                mlflow.log_params(gs.best_params_)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(
                    pipeline,
                    name="model",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )
                mlflow.log_artifact(str(plot_confusion(y_test, y_pred, name)))
                fi_path = plot_feature_importance(pipeline, feature_names, name)
                if fi_path:
                    mlflow.log_artifact(str(fi_path))

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            rocs[name] = (fpr, tpr, metrics["roc_auc"])

            if metrics["roc_auc"] > best["score"]:
                best.update(
                    score=metrics["roc_auc"],
                    name=name,
                    pipeline=pipeline,
                    params=gs.best_params_,
                    metrics=metrics,
                )

    plot_roc_curves(rocs, FIG_DIR / "05_roc_curves.png")

    # Persist a SINGLE self-contained pipeline artifact (preprocess + model).
    joblib.dump(best["pipeline"], MODELS_DIR / "best_pipeline.pkl")

    summary = {
        "best_model": best["name"],
        "best_params": best["params"],
        "best_metrics": best["metrics"],
        "all_results": results,
    }
    resolve(CONFIG["paths"]["metrics_file"]).write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== TRAINING SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))

    if use_mlflow:
        with mlflow.start_run(run_name=f"best_{best['name']}_registry"):
            mlflow.log_params(best["params"])
            mlflow.log_metrics(best["metrics"])
            mlflow.sklearn.log_model(
                best["pipeline"],
                name="model",
                registered_model_name=CONFIG["mlflow"]["registered_model_name"],
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

    return summary


class _NullCtx:
    """No-op context manager for running without MLflow."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Train diabetes risk models.")
    p.add_argument(
        "--data",
        type=str,
        default=str(resolve(CONFIG["data"]["raw_path"])),
        help="Path to raw CSV.",
    )
    p.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train_all(Path(args.data), use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
