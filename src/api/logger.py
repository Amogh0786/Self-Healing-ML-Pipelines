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
                    request_id TEXT UNIQUE,
                    timestamp TEXT NOT NULL,
                    MedInc REAL NOT NULL,
                    HouseAge REAL NOT NULL,
                    AveRooms REAL NOT NULL,
                    AveBedrms REAL NOT NULL,
                    Population REAL NOT NULL,
                    AveOccup REAL NOT NULL,
                    Latitude REAL NOT NULL,
                    Longitude REAL NOT NULL,
                    prediction REAL NOT NULL,
                    actual_value REAL
                )
            """)
            conn.commit()
            conn.close()

    def log_batch(self, batch: list) -> None:
        """
        Efficiently logs a batch of events (predictions and feedback) to SQLite.
        """
        predictions = []
        feedback = []
        
        for event in batch:
            if event["type"] == "prediction":
                timestamp = datetime.utcnow().isoformat()
                predictions.append((
                    event["request_id"], timestamp,
                    event["features"]["MedInc"], event["features"]["HouseAge"],
                    event["features"]["AveRooms"], event["features"]["AveBedrms"],
                    event["features"]["Population"], event["features"]["AveOccup"],
                    event["features"]["Latitude"], event["features"]["Longitude"],
                    event["prediction"]
                ))
            elif event["type"] == "feedback":
                feedback.append((event["actual_value"], event["request_id"]))
                
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if predictions:
                cursor.executemany("""
                    INSERT INTO inference_logs (
                        request_id, timestamp, MedInc, HouseAge, AveRooms, AveBedrms,
                        Population, AveOccup, Latitude, Longitude, prediction
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, predictions)
                
            if feedback:
                cursor.executemany("""
                    UPDATE inference_logs SET actual_value = ? WHERE request_id = ?
                """, feedback)
                
            conn.commit()
            conn.close()

    def get_log_count(self) -> int:
        """Returns total number of logged inference requests."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM inference_logs")
            count = cursor.fetchone()[0]
            conn.close()
            return int(count)
