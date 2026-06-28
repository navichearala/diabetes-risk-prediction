# AWS Cloud Integration — Design Notes

These are planning notes for running the **Diabetes Risk Prediction** workflow
on AWS. They map each local step (EDA → preprocess → train → track → serve) onto
managed AWS services, with three maturity tiers so the project can grow from a
weekend portfolio deploy into a production MLOps stack.

> Status: design / "noting points" only. No infrastructure is provisioned by
> this repo. Treat as an architecture reference.

---

## 1. Service Mapping (local → AWS)

| Local component                | AWS service                                  | Notes |
|--------------------------------|----------------------------------------------|-------|
| `data/diabetes_raw.csv`        | **Amazon S3** (`s3://<bucket>/raw/`)          | Versioned bucket; raw + processed prefixes |
| `data_preprocessing.py`        | **SageMaker Processing Job** or AWS Glue      | Same sklearn `Pipeline` packaged in a container |
| `train.py` (GridSearchCV)      | **SageMaker Training Job**                    | Spot instances for cost; pass `config.yaml` as a hyperparameter file |
| MLflow tracking (`mlflow.db`)  | **SageMaker Experiments** or MLflow on Fargate + RDS/S3 | RDS Postgres = backend store, S3 = artifact store |
| Model registry                 | **SageMaker Model Registry**                 | Replaces the local MLflow registry; supports approval gates |
| `best_pipeline.pkl`            | **S3** + registered model package            | Self-contained pipeline = one artifact, no scaler drift |
| `app.py` (FastAPI)             | **Lambda + API Gateway** *or* **ECS Fargate** *or* **SageMaker Endpoint** | Pick by latency/scale needs (see §4) |
| `reports/figures/`             | **S3** (static) + optional CloudFront         | Share EDA/ROC plots |
| CI (`.github/workflows`)       | **GitHub Actions → OIDC → AWS**               | No long-lived AWS keys; assume an IAM role |
| Orchestration                  | **Step Functions** or SageMaker Pipelines     | Chains process → train → evaluate → register |
| Monitoring                     | **CloudWatch** + **SageMaker Model Monitor**  | Latency, errors, data/feature drift |
| Secrets/config                 | **SSM Parameter Store** / **Secrets Manager** | Store `config.yaml` values, DB creds |

---

## 2. Reference Architecture (target state)

```text
        ┌─────────────┐     ┌──────────────────────┐
Raw CSV │  Amazon S3  │────▶│ SageMaker Processing  │  (clean + feature eng.)
        │  raw/ proc/ │     └──────────┬───────────┘
        └─────────────┘                │ processed data → S3
                                        ▼
                            ┌──────────────────────┐
                            │ SageMaker Training Job│  (GridSearchCV, CV)
                            └──────────┬───────────┘
                                       │ metrics → Experiments
                                       │ pipeline.pkl → S3
                                       ▼
                            ┌──────────────────────┐
                            │  Model Registry       │  (versioned, approval)
                            └──────────┬───────────┘
                          approve      │
                                       ▼
        ┌───────────────────────────────────────────────┐
        │  Serving (one of):                              │
        │   • Lambda + API Gateway   (spiky / cheap)      │
        │   • SageMaker Endpoint     (managed, autoscale) │
        │   • ECS Fargate (FastAPI)  (full control)       │
        └───────────────────┬───────────────────────────┘
                            ▼
              CloudWatch + SageMaker Model Monitor
              (latency, errors, data drift alarms)

  Orchestrated by Step Functions / SageMaker Pipelines.
  CI/CD: GitHub Actions → OIDC role → ECR / SageMaker / Lambda.
```

---

## 3. Three Implementation Tiers

### Tier 1 — Minimal / portfolio (lowest cost)
- Store the raw CSV and `best_pipeline.pkl` in **S3**.
- Containerize `app.py`, deploy to **AWS Lambda** (container image) behind
  **API Gateway**. Lambda pulls the pipeline from S3 on cold start (cache in `/tmp`).
- Train locally or in a one-off **EC2** spot box; upload the artifact to S3.
- Cost: pay-per-request; near-zero when idle.

### Tier 2 — Managed ML (recommended)
- **SageMaker Training Job** runs `train.py` on a managed instance; reads
  `config.yaml`, writes the pipeline + metrics to S3.
- Register the model in **SageMaker Model Registry** with a manual approval gate.
- Deploy the approved version to a **SageMaker Real-Time Endpoint** (autoscaling).
- **SageMaker Experiments** replaces local MLflow for tracking.

### Tier 3 — Full MLOps / production
- **SageMaker Pipelines** (or Step Functions) orchestrate
  process → train → evaluate → conditional-register on ROC-AUC threshold.
- **SageMaker Model Monitor** schedules data-quality + drift checks against a
  captured baseline; alarms via CloudWatch + SNS.
- Automated retraining trigger (EventBridge schedule or drift alarm).
- Blue/green or canary endpoint deployments.
- Full IaC via **Terraform** or **AWS CDK**.

---

## 4. Serving Option Trade-offs

| Option              | Best for                      | Pros                          | Cons |
|---------------------|-------------------------------|-------------------------------|------|
| Lambda + API GW     | Spiky, low-volume traffic     | Cheap idle, zero servers      | Cold starts, 250 MB unzipped image limit, 15-min cap |
| SageMaker Endpoint  | Steady ML traffic             | Autoscale, built-in monitor   | Always-on instance cost |
| ECS Fargate (API)   | Custom runtime / full control | Reuses `app.py` as-is, no cold start | You manage scaling + LB |

For this model (tiny sklearn pipeline, sub-ms inference), **Lambda + API Gateway**
is the most cost-effective starting point.

---

## 5. CI/CD with GitHub Actions → AWS

- Use **OIDC federation**: GitHub Actions assumes an IAM role, no static keys.
- On merge to `main`:
  1. Run existing CI (lint + tests + smoke train).
  2. Build the serving container, push to **Amazon ECR**.
  3. Trigger a SageMaker training/pipeline run (or update the Lambda image).
  4. On ROC-AUC ≥ threshold, register + (optionally auto-)approve the model.
- Keep `config.yaml` thresholds in the repo; mirror runtime secrets in SSM.

---

## 6. Cost & Security Checklist

- **Cost:** prefer spot for training; scale endpoints to zero where possible
  (Lambda or SageMaker Serverless Inference); set S3 lifecycle rules; tag all
  resources by project for cost allocation.
- **Security:** least-privilege IAM per job; encrypt S3 (SSE-KMS) and endpoints;
  VPC-isolate training/serving; no PHI in logs (this is healthcare data — treat
  as sensitive even though the Pima dataset is public/de-identified).
- **Reproducibility:** pin the training image digest; log the `config.yaml`
  hash and dataset version (S3 version ID) with each model in the registry.

---

## 7. Mapping to This Repo's Files

| Repo file              | AWS migration action |
|------------------------|----------------------|
| `config.yaml`          | Pass as hyperparameters / SSM params to jobs |
| `src/data_preprocessing.py` | Package into the Processing/Training container |
| `src/train.py`         | Entry point for the SageMaker Training Job |
| `src/predict.py` / `app.py` | Inference handler for Lambda / Endpoint / Fargate |
| `best_pipeline.pkl`    | Upload to S3, register in Model Registry |
| `.github/workflows/ci.yml` | Extend with build-push-deploy + OIDC role |
| `tests/`               | Run unchanged as the CI quality gate |
