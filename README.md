# Diabetes Risk Prediction — End-to-End Healthcare ML Project

An end-to-end machine learning workflow that predicts the likelihood of diabetes
in patients using clinical and demographic features. Built as an intermediate-level
portfolio project covering data ingestion, EDA, feature engineering, model training,
hyperparameter tuning, evaluation, MLOps tracking, and inference.

**Domain:** Healthcare / Clinical Risk Stratification
**Dataset:** Pima Indians Diabetes Database (768 patients, 8 clinical features)
**Best Model:** Logistic Regression — ROC-AUC **0.839**, Accuracy **74.0%**

---

## Project Structure

```
diabetes-risk-prediction/
├── config.yaml                       # Single source of truth (paths, seeds, grids)
├── data/
│   └── diabetes_raw.csv              # Raw Pima Indians Diabetes data
├── src/
│   ├── config.py                     # Typed config loader
│   ├── data_preprocessing.py         # Cleaning, feature eng., sklearn Pipeline
│   ├── eda.py                        # Exploratory analysis + visualizations
│   ├── train.py                      # Multi-model training (CLI) + MLflow
│   ├── predict.py                    # Inference / batch scoring (CLI)
│   └── app.py                        # FastAPI serving layer (/predict, /health)
├── models/
│   └── best_pipeline.pkl             # Self-contained pipeline (preprocess + model)
├── reports/
│   ├── figures/                      # Saved PNG visualizations
│   └── metrics.json                  # Final metrics summary
├── tests/
│   └── test_pipeline.py              # Pytest suite
├── docs/
│   └── AWS_INTEGRATION.md            # AWS cloud migration design notes
├── .github/workflows/ci.yml          # CI: lint + format + tests + smoke train
├── Makefile                          # make install / eda / train / test / lint
├── pyproject.toml                    # ruff + black + pytest config
├── conftest.py                       # Puts src/ on the test path
├── mlruns/ + mlflow.db               # MLflow tracking + model registry (gitignored)
├── requirements.txt
└── README.md
```

### What changed in v2 (workflow efficiency)
- **Single config file** (`config.yaml`) — no magic numbers scattered across modules.
- **One self-contained `Pipeline`** bundles imputation + scaling + model, so the
  exact transforms learned in training are reused at inference. This removed a
  real bug: the old `predict.py` hardcoded imputation medians that could drift
  from the training data, and the separate `scaler.pkl` is no longer needed.
- **GridSearchCV runs over the pipeline**, preventing preprocessing data leakage.
- **CLI flags** on `train.py` / `predict.py`; **Makefile** one-liners.
- **GitHub Actions CI** runs lint (ruff), format check (black), tests, and a
  smoke training run on every push/PR.
- **FastAPI app** (`app.py`) for online scoring, sharing one code path with batch.
- **AWS migration notes** in `docs/AWS_INTEGRATION.md`.

---

## Workflow

### 1. Data Ingestion & Cleaning
- Loaded 768 patient records with 8 clinical features.
- Treated clinically-impossible zeros in `Glucose`, `BloodPressure`, `SkinThickness`,
  `Insulin`, and `BMI` as missing values and imputed with the median.

### 2. Feature Engineering
Three domain-driven categorical features were derived from continuous variables:
- **BMI_Category** — Underweight / Normal / Overweight / Obese
- **Age_Group** — Young / Adult / Middle / Senior
- **Glucose_Risk** — Normal / Prediabetic / Diabetic / Severe

One-hot encoded, then split 80/20 with stratification. Imputation (median) and
`StandardScaler` live inside the sklearn `Pipeline`, fit only on the training fold.

### 3. Exploratory Data Analysis
Five visualizations generated in `reports/figures/`:
1. Target distribution (268 diabetic vs 500 non-diabetic)
2. Feature correlation heatmap
3. Feature distributions by outcome (KDE plots)
4. Glucose vs BMI scatter with clinical thresholds
5. Combined ROC curves for all models

### 4. Model Training & Hyperparameter Tuning
Three classifiers trained with **5-fold stratified cross-validation** and
`GridSearchCV` over ROC-AUC:

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------------|----------|-----------|--------|--------|---------|
| Logistic Regression | 0.7403   | 0.6400    | 0.5926 | 0.6154 | **0.8391** |
| Random Forest       | 0.7597   | 0.6809    | 0.5926 | 0.6337 | 0.8259 |
| Gradient Boosting   | 0.7532   | 0.6739    | 0.5741 | 0.6200 | 0.8259 |

Logistic Regression won on ROC-AUC and was chosen as the production model.

### 5. MLOps — MLflow Tracking & Model Registry
Every run logs hyperparameters, metrics, confusion matrix and feature-importance
artifacts to a local SQLite-backed MLflow store. The best model is registered as
`DiabetesReadmissionBest` (version 1) in the MLflow Model Registry.

Launch the UI locally:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 6. Inference
The saved `best_pipeline.pkl` applies preprocessing + model in one step:
```python
from src.predict import predict

predict([{
    "Pregnancies": 2, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
    "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50,
}])
# -> [{"diabetes_risk": 0.76, "prediction": 1, "risk_band": "High"}]
```

Risk bands: **Low** (<0.30) · **Moderate** (0.30–0.60) · **High** (≥0.60).

---

## Quickstart

```bash
git clone https://github.com/navichearala/diabetes-risk-prediction.git
cd diabetes-risk-prediction
pip install -r requirements.txt

# Via Makefile
make eda            # generate visualizations
make train          # train all models + log to MLflow
make predict        # run sample inference
make test           # run unit tests
make lint           # ruff + black --check

# Or directly (with CLI options)
python src/train.py --no-mlflow            # train without MLflow tracking
python src/predict.py --json patients.json # batch-score a JSON file
```

### Serve the model as an API
```bash
uvicorn app:app --reload --app-dir src
# POST a list of patient records to http://127.0.0.1:8000/predict
```

---

## Cloud Deployment
See [`docs/AWS_INTEGRATION.md`](docs/AWS_INTEGRATION.md) for design notes on
running this workflow on AWS (S3, SageMaker, Lambda/API Gateway, Step Functions,
Model Registry, and CI/CD via GitHub Actions OIDC), with three maturity tiers.

---

## Tech Stack
- **Python 3.10+**, **pandas**, **NumPy**, **scikit-learn**
- **matplotlib**, **seaborn** for visualization
- **MLflow** for experiment tracking and model registry
- **FastAPI** + **uvicorn** for the serving layer
- **pytest** for unit testing, **ruff** + **black** for lint/format
- **joblib** for model serialization, **PyYAML** for config

---

## Business Value
- Helps clinicians flag high-risk patients early for preventive intervention.
- Reduces downstream costs of unmanaged diabetes (hospitalization, complications).
- Provides interpretable risk bands rather than opaque probabilities.

---

## Author
**Naveen Chearala** — AI/ML Analyst
Chicago, IL · [GitHub](https://github.com/navichearala)
