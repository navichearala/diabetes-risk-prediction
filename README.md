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
diabetes_readmission_prediction/
├── data/
│   └── diabetes_raw.csv              # Raw Pima Indians Diabetes data
├── notebooks/                         # Jupyter exploration (optional)
├── src/
│   ├── data_preprocessing.py         # Cleaning, imputation, feature engineering
│   ├── eda.py                        # Exploratory analysis + visualizations
│   ├── train.py                      # Multi-model training with MLflow tracking
│   └── predict.py                    # Inference / batch scoring
├── models/
│   ├── best_model.pkl                # Final serialized model
│   └── scaler.pkl                    # Fitted StandardScaler
├── reports/
│   ├── figures/                      # Saved PNG visualizations
│   └── metrics.json                  # Final metrics summary
├── tests/
│   └── test_pipeline.py              # Pytest suite
├── mlruns/ + mlflow.db               # MLflow tracking + model registry
├── requirements.txt
└── README.md
```

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

One-hot encoded, scaled with `StandardScaler`, then split 80/20 with stratification.

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

python src/eda.py          # generate visualizations
python src/train.py        # train all models + log to MLflow
python src/predict.py      # run sample inference
pytest tests/              # run unit tests
```

---

## Tech Stack
- **Python 3.10+**, **pandas**, **NumPy**, **scikit-learn**
- **matplotlib**, **seaborn** for visualization
- **MLflow** for experiment tracking and model registry
- **pytest** for unit testing
- **joblib** for model serialization

---

## Business Value
- Helps clinicians flag high-risk patients early for preventive intervention.
- Reduces downstream costs of unmanaged diabetes (hospitalization, complications).
- Provides interpretable risk bands rather than opaque probabilities.

---

## Author
**Naveen Chearala** — AI/ML Analyst
Chicago, IL · [GitHub](https://github.com/navichearala)
