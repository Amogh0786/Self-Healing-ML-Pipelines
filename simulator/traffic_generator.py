"""
Traffic Generator and Synthetic Data Drift Injection script.
Simulates normal live API requests and injects skewed feature distributions to trigger drift alerts.
"""
import random
import time
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import requests
from sklearn.datasets import fetch_california_housing
from src.api.logger import InferenceLogger
from src.model.train_baseline import FEATURE_NAMES


def generate_single_payload(
    base_df: pd.DataFrame,
    drift: bool = False
) -> Dict[str, float]:
    """
    Generates a single housing feature payload.
    If drift=True, artificially skews MedInc (+3.5x) and HouseAge (+20 years)
    to mathematically violate baseline assumptions (high KL divergence).
    """
    row = base_df.sample(1).iloc[0]
    payload = {
        col: float(row[col]) for col in FEATURE_NAMES
    }
    
    if drift:
        # Inject significant statistical drift
        payload["MedInc"] = float(payload["MedInc"] * 3.5 + 5.0)
        payload["HouseAge"] = float(min(100.0, payload["HouseAge"] + 25.0))
        payload["AveRooms"] = float(payload["AveRooms"] * 1.8)

    return payload


def generate_traffic(
    num_requests: int = 50,
    drift: bool = False,
    api_url: Optional[str] = "http://localhost:8000/predict",
    db_path: str = "logs.db",
    delay_seconds: float = 0.0
) -> int:
    """
    Sends simulated traffic to the API endpoint or logs directly to SQLite if offline.
    
    Args:
        num_requests: Number of payloads to send.
        drift: Whether to inject data drift.
        api_url: FastAPI endpoint URL. If unreachable, falls back to direct logging.
        db_path: SQLite path for direct logging fallback.
        delay_seconds: Sleep delay between requests.
        
    Returns:
        count_sent: Number of requests successfully sent/logged.
    """
    base_data = fetch_california_housing(as_frame=True)
    df = base_data.frame
    logger = InferenceLogger(db_path=db_path)
    
    use_http = False
    if api_url:
        try:
            # Test connectivity
            resp = requests.get(api_url.replace("/predict", "/health"), timeout=1)
            use_http = resp.status_code == 200
        except Exception:
            use_http = False

    print(f"[{'DRIFTED' if drift else 'NORMAL'} TRAFFIC] Sending {num_requests} requests via {'HTTP API' if use_http else 'Direct DB Logging'}...")

    count_sent = 0
    for _ in range(num_requests):
        payload = generate_single_payload(df, drift=drift)
        
        if use_http and api_url:
            try:
                r = requests.post(api_url, json=payload, timeout=2)
                if r.status_code == 200:
                    count_sent += 1
            except Exception:
                # Fallback to direct logging
                logger.log_request(payload, prediction=0.0)
                count_sent += 1
        else:
            # Simulate a reasonable dummy prediction for offline logging
            dummy_pred = payload["MedInc"] * 0.45 + random.uniform(0.5, 1.5)
            logger.log_request(payload, prediction=dummy_pred)
            count_sent += 1

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    print(f"[{'DRIFTED' if drift else 'NORMAL'} TRAFFIC] Completed {count_sent}/{num_requests} requests.")
    return count_sent


def simulate_drifted_requests(num_requests: int = 100, db_path: str = "logs.db") -> int:
    """Convenience wrapper for generating drifted traffic."""
    return generate_traffic(num_requests=num_requests, drift=True, api_url=None, db_path=db_path)
