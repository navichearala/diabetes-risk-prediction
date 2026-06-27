# Project Summary

## Objective
Predict diabetes risk in patients from clinical and demographic features so that
healthcare providers can prioritize preventive care for high-risk individuals.

## Data
- **Source:** Pima Indians Diabetes Database (UCI / public mirror)
- **Records:** 768 female patients (≥21 yrs, Pima Indian heritage)
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
  DiabetesPedigreeFunction, Age
- **Target:** Outcome (1 = diabetic within 5 yrs, 0 = not)

## Approach
1. **Cleaning** — replaced impossible zeros with NaN, imputed with median.
2. **Feature engineering** — derived `BMI_Category`, `Age_Group`, `Glucose_Risk`.
3. **Scaling & split** — `StandardScaler`, 80/20 stratified split.
4. **Modeling** — Logistic Regression, Random Forest, Gradient Boosting with
   `GridSearchCV` and 5-fold stratified CV on ROC-AUC.
5. **Tracking** — MLflow logging of params, metrics, artifacts, plus model
   registry (`DiabetesReadmissionBest`).
6. **Testing** — pytest suite covering load / clean / feature-engineer / prepare.
7. **Inference** — `predict.py` returns probability, hard prediction, and
   clinical risk band (Low / Moderate / High).

## Final Results
| Metric    | Score |
|-----------|-------|
| ROC-AUC   | 0.839 |
| Accuracy  | 0.740 |
| Precision | 0.640 |
| Recall    | 0.593 |
| F1        | 0.615 |

**Winning model:** Logistic Regression (`C=0.1`, `penalty=l2`).
Logistic regression was selected over tree ensembles for its higher ROC-AUC
and superior interpretability — important for clinical adoption.

## Why this matters
- **Clinical interpretability** beats marginal accuracy gains in healthcare.
- **End-to-end pipeline** — reproducible from raw CSV to registered model.
- **MLOps-ready** — MLflow registry enables versioning and rollback.
