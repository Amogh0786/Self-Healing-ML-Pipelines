"""
FastAPI Serving Application for California Housing Price Prediction.
Includes model serving, real-time database logging, health probes, and model hot-reload.
"""
import os
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from src.api.schemas import HousingFeatures, PredictionResponse
from src.api.logger import InferenceLogger
from src.model.train_baseline import FEATURE_NAMES

app = FastAPI(
    title="Self-Healing MLOps Inference Service",
    description="Production California Housing regression API with continuous drift logging",
    version="1.0.0"
)

# Global state
model = None
model_version_tag = "Production-v1"
logger_service = InferenceLogger(db_path="logs.db")


def load_model() -> None:
    """Loads the production model from disk."""
    global model, model_version_tag
    candidates = ["production_model.joblib", "baseline_model.joblib"]
    for path in candidates:
        if os.path.exists(path):
            model = joblib.load(path)
            model_version_tag = f"Production-{os.path.basename(path).replace('.joblib', '')}"
            print(f"Loaded serving model from {path} ({model_version_tag})")
            return
    print("Warning: No pre-trained model file found on disk.")


@app.on_event("startup")
def startup_event():
    load_model()


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures):
    """
    Predicts median house value and logs incoming features to SQLite for drift evaluation.
    """
    global model, model_version_tag
    if model is None:
        load_model()
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded."
            )

    feature_dict = features.to_dict()
    df = pd.DataFrame([feature_dict])[FEATURE_NAMES]

    try:
        prediction = float(model.predict(df)[0])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

    # Log feature vector and prediction asynchronously/thread-safely
    logger_service.log_request(feature_dict, prediction)

    return PredictionResponse(
        prediction=prediction,
        model_version=model_version_tag,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/health")
def health_check():
    """Liveness/readiness probe endpoint for Kubernetes Rolling Updates."""
    is_ready = model is not None
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": is_ready,
        "model_version": model_version_tag
    }


@app.get("/metrics")
def get_metrics():
    """Returns inference activity metrics for observability."""
    return {
        "total_logged_requests": logger_service.get_log_count(),
        "model_version": model_version_tag,
        "status": "serving" if model is not None else "down"
    }


@app.post("/reload_model")
def reload_model_endpoint():
    """Hot-reloads the production model after automated retraining and promotion."""
    load_model()
    return {
        "status": "reloaded",
        "model_version": model_version_tag
    }
