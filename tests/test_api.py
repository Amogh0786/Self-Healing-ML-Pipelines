"""
Unit tests for FastAPI serving endpoints and database request logging.
"""
import os
import pytest
from fastapi.testclient import TestClient
from src.api.app import app, logger_service


client = TestClient(app)


def test_health_check():
    """Tests the /health liveness probe endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_metrics_endpoint():
    """Tests the /metrics endpoint for observability stats."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_logged_requests" in data
    assert "model_version" in data


def test_prediction_logging():
    """Tests /predict endpoint returns valid prediction with request_id."""
    # Ensure baseline model is trained or create dummy model for test
    from src.model.train_baseline import train_baseline_model
    if not os.path.exists("production_model.joblib"):
        train_baseline_model(n_estimators=5, max_depth=3)
        
    payload = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Error response: {response.text}"
    
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert "request_id" in data
    assert isinstance(data["request_id"], str)
    assert len(data["request_id"]) > 0


def test_feedback_endpoint():
    """Tests POST /feedback stores ground truth without error."""
    from src.model.train_baseline import train_baseline_model
    if not os.path.exists("production_model.joblib"):
        train_baseline_model(n_estimators=5, max_depth=3)

    # First get a request_id
    payload = {
        "MedInc": 5.0, "HouseAge": 25.0, "AveRooms": 5.0,
        "AveBedrms": 1.0, "Population": 500.0, "AveOccup": 2.5,
        "Latitude": 34.0, "Longitude": -118.0
    }
    predict_resp = client.post("/predict", json=payload)
    assert predict_resp.status_code == 200
    request_id = predict_resp.json()["request_id"]

    # Submit feedback with actual value
    feedback_resp = client.post("/feedback", json={
        "request_id": request_id,
        "actual_value": 2.5
    })
    assert feedback_resp.status_code == 200
    assert feedback_resp.json()["status"] == "feedback_queued"
