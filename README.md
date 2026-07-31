# Self-Healing Automated Production MLOps Pipeline

[![CI/CD Pipeline](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![ISO 27001 Ready](https://img.shields.io/badge/Audit-ISO%2027001%20Ready-purple.svg)]()

An enterprise-grade, closed-loop machine learning pipeline that moves a model from a static notebook into an automated, self-healing production system with deterministic guardrails and contextual bandit recovery policies.

```
  +-------------------------------------------------------------------------------+
  |                              CLOSED-LOOP MLOPS                                |
  |                                                                               |
  |    +---------------+        Predictions &        +-----------------------+    |
  |    |  FastAPI App  |---------------------------->| Inference Request Log |    |
  |    | (Production)  |        Feature Vectors      +-----------------------+    |
  |    +---------------+                                         |                |
  |           ^                                                  |                |
  |           | Zero-Downtime                                    v                |
  |           | Kubernetes                           +-----------------------+    |
  |           | Rolling Update                       | KL Divergence & PSI   |    |
  |    +---------------+                             |   Drift Detector      |    |
  |    | Kubernetes    |                             +-----------------------+    |
  |    | Deployment    |                                         |                |
  |    +---------------+                                         v                |
  |           ^                                      +-----------------------+    |
  |           | Promote Model if                     | Hybrid Healing Engine |    |
  |           | Candidate Outperforms                | (Rules + Bandits)     |    |
  |           |                                      +-----------------------+    |
  |           |                                        |     |          |         |
  |           |                              [RETRAIN] |     | [ROLLBACK]         |
  |           |                                        v     v          |         |
  |    +---------------+                             +-----------------------+    |
  |    | MLflow Model  |<----------------------------| Apache Airflow        |    |
  |    |   Registry    |       Log Staging           | Branching Controller  |    |
  |    +---------------+       Model & Metrics       +-----------------------+    |
  +-------------------------------------------------------------------------------+
```

---

## Architectural Breakdown

### 1. Inference & Logging (`src/api/` & `src/model/`)
- **FastAPI Service**: Serves real-time predictions via `/predict`, `/health`, and `/metrics`.
- **Database Request Logging**: Asynchronously logs incoming feature vectors and model predictions to a persistent SQL database (`logs.db` / PostgreSQL) for continuous drift monitoring.
- **Baseline Training (`src/model/train_baseline.py`)**: Trains a Random Forest regressor on the California Housing dataset, logs hyperparameters and metrics to MLflow, and serializes the training feature probability distributions $P(x)$ as a reference baseline.

### 2. Quantifying the Trigger: Mathematical Drift Detection (`src/drift/`)
To mathematically prove that live data distribution $Q(x)$ has deviated from the training baseline $P(x)$, we compute the **Kullback-Leibler (KL) Divergence**:

\[D_{\text{KL}}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \ln\left(\frac{P(x)}{Q(x)}\right)\]

- **Quantile/Histogram Binning with Laplace Smoothing**: Handles zero-probability bins without numerical overflow.
- **Population Stability Index (PSI)**: Also computed as a symmetrical drift verification metric:
  \[\text{PSI} = \sum_{i} \left(Q_i - P_i\right) \ln\left(\frac{Q_i}{P_i}\right)\]
- **Alert Trigger**: When $D_{\text{KL}} > \tau$ (default threshold $\tau = 0.15$), a retraining alert is raised.

### 3. Hybrid Healing Decision Engine (`src/healing/`)
Rather than naive binary retraining toggles, the system uses a **3-Layer Hybrid Control Loop**:
- **Deterministic Safety Guardrails (`src/healing/rules_engine.py`)**:
  - Enforces configurable cooldown timers (default 30 minutes) to prevent sequential retraining loops.
  - Triggers critical emergency overrides if severe drift ($D_{\text{KL}} \ge 0.80$) is observed.
- **Contextual Bandit / Bayesian Policy (`src/healing/bandit_policy.py`)**:
  - Evaluates multi-attribute context vectors `(max_kl, max_psi, sample_count, error_rate)`.
  - Calculates expected recovery utility across three candidate arms: **`RETRAIN`**, **`ROLLBACK`**, and **`FALLBACK`**.
- **Audit Logger (`src/healing/decision_engine.py`)**:
  - Records ISO 27001-ready JSON audit trails (`healing_audit_log.json`) with confidence scores and mathematical rationales.

### 4. Orchestration with Apache Airflow (`airflow_pipeline/dags/`)
- **`self_healing_mlops_dag.py`**: An automated DAG scheduled daily (or triggerable via webhook).
- Uses `BranchPythonOperator` to evaluate drift and route to the appropriate recovery branch (`RETRAIN`, `ROLLBACK`, `FALLBACK`, or `NONE`).

### 5. MLflow Governance & Registry (`src/mlflow_utils/`)
- **Model Registry**: Keeps an immutable ledger of all model iterations.
- **Automated Promotion**: Evaluates candidate (`Staging`) model against the current live (`Production`) model on a holdout test set. Promotes the new model only if its $R^2$ score is higher and RMSE is lower.

### 6. Docker Containerization & Kubernetes Zero-Downtime Deployment (`docker/` & `k8s/`)
- **Dockerfiles**:
  - `Dockerfile.api`: Production multi-stage image for the FastAPI serving application.
  - `Dockerfile.trainer`: Image for training, drift detection, and evaluation tasks.
- **Kubernetes Manifests**:
  - Configures `RollingUpdate` strategy (`maxUnavailable: 0`, `maxSurge: 1`) with `readinessProbe` and `livenessProbe` to ensure zero-downtime model swaps.
- **Automated Rollout (`src/deploy/k8s_rollout.py`)**: Executes declarative deployment updates to switch traffic to the newly promoted container image.

---

## Quickstart & Demo

### Running the End-to-End Simulation Demo
We provide a standalone CLI interactive demo runner (`demo_pipeline.py`) that executes the complete closed-loop workflow:
1. Trains the baseline Random Forest model and stores baseline distributions.
2. Boots up the FastAPI inference service.
3. Simulates normal incoming API requests and verifies `drift == False`.
4. Injects skewed/drifted feature requests (`MedInc` shifted by 3x) and verifies `drift == True`.
5. Executes the automated retraining and MLflow evaluation pipeline.
6. Promotes the winning candidate model and triggers a simulated Kubernetes rolling deployment.

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run the end-to-end self-healing demo
python demo_pipeline.py
```

### Running Automated Tests
```bash
pytest tests/ -v
```
