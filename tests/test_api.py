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
    """Tests /predict endpoint and verifies request is logged to DB."""
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
    
    count_before = logger_service.get_log_count()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Error response: {response.text}"
    
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    
    count_after = logger_service.get_log_count()
    assert count_after == count_before + 1, "Inference logger must increment request count by 1"
