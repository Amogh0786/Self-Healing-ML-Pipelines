"""
FastAPI Serving Application for California Housing Price Prediction.
Includes model serving, real-time database logging, health probes, and model hot-reload.
"""
import os
import uuid
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from src.api.schemas import HousingFeatures, PredictionResponse, FeedbackRequest
from src.api.logger import InferenceLogger
from src.model.train_baseline import FEATURE_NAMES
from src.api.event_stream import producer, EventStreamConsumer

app = FastAPI(
    title="Self-Healing MLOps Inference Service",
    description="Production California Housing regression API with continuous drift logging",
    version="1.0.0"
)

# Global state
model = None
shadow_model = None
model_version_tag = "Production-v1"
logger_service = InferenceLogger(db_path="logs.db")
consumer = EventStreamConsumer(producer, logger_service)

def load_model() -> None:
    """Loads the production model and optional shadow/staging model from disk."""
    global model, shadow_model, model_version_tag
    candidates = ["production_model.joblib", "baseline_model.joblib"]
    for path in candidates:
        if os.path.exists(path):
            model = joblib.load(path)
            model_version_tag = f"Production-{os.path.basename(path).replace('.joblib', '')}"
            print(f"Loaded serving model from {path} ({model_version_tag})")
            break
            
    # Attempt to load Shadow (Staging) Model
    if os.path.exists("staging_model.joblib"):
        try:
            shadow_model = joblib.load("staging_model.joblib")
            print("Loaded Shadow Model for silent A/B evaluation.")
        except Exception:
            shadow_model = None
    else:
        shadow_model = None

    if model is None:
        print("Warning: No pre-trained model file found on disk.")

@app.on_event("startup")
def startup_event():
    load_model()
    consumer.start()

@app.on_event("shutdown")
def shutdown_event():
    consumer.stop()

def run_shadow_scoring(df: pd.DataFrame, prod_prediction: float):
    """Silently predicts using the staging model and logs the deviation."""
    if shadow_model is not None:
        try:
            shadow_pred = float(shadow_model.predict(df)[0])
            deviation = abs(prod_prediction - shadow_pred)
            if deviation > 1.0: # Huge deviation
                print(f"[SHADOW ALERT] Staging model differs significantly! Prod: {prod_prediction:.2f}, Shadow: {shadow_pred:.2f}")
        except Exception:
            pass

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures, background_tasks: BackgroundTasks):
    """
    Predicts median house value and asynchronously pushes event to EventStream.
    Executes shadow scoring if a staging model is active.
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

    request_id = str(uuid.uuid4())
    
    # Non-blocking async event publish
    producer.publish_prediction_event(request_id, feature_dict, prediction)
    
    # Trigger shadow deployment scoring
    background_tasks.add_task(run_shadow_scoring, df, prediction)

    return PredictionResponse(
        request_id=request_id,
        prediction=prediction,
        model_version=model_version_tag,
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """
    Ingests delayed ground truth labels (Concept Drift) to inform the Hybrid Healing Engine.
    """
    producer.publish_feedback_event(feedback.request_id, feedback.actual_value)
    return {"status": "feedback_queued", "request_id": feedback.request_id}


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
