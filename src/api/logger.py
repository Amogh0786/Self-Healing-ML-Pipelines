"""
Database logging service for inference requests and predictions.
Logs feature vectors for continuous drift monitoring.
"""
import os
import sqlite3
import threading
from datetime import datetime
from typing import Dict, Any


class InferenceLogger:
    """
    Thread-safe SQLite logging service for storing live inference requests.
    """
    def __init__(self, db_path: str = "logs.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    MedInc REAL NOT NULL,
                    HouseAge REAL NOT NULL,
                    AveRooms REAL NOT NULL,
                    AveBedrms REAL NOT NULL,
                    Population REAL NOT NULL,
                    AveOccup REAL NOT NULL,
                    Latitude REAL NOT NULL,
                    Longitude REAL NOT NULL,
                    prediction REAL NOT NULL
                )
            """)
            conn.commit()
            conn.close()

    def log_request(self, features: Dict[str, float], prediction: float) -> int:
        """
        Logs a single inference feature vector and prediction to SQLite.
        
        Returns:
            row_id: Inserted record ID.
        """
        timestamp = datetime.utcnow().isoformat()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO inference_logs (
                    timestamp, MedInc, HouseAge, AveRooms, AveBedrms,
                    Population, AveOccup, Latitude, Longitude, prediction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                features["MedInc"],
                features["HouseAge"],
                features["AveRooms"],
                features["AveBedrms"],
                features["Population"],
                features["AveOccup"],
                features["Latitude"],
                features["Longitude"],
                prediction
            ))
            row_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return row_id or 0

    def get_log_count(self) -> int:
        """Returns total number of logged inference requests."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM inference_logs")
            count = cursor.fetchone()[0]
            conn.close()
            return int(count)
