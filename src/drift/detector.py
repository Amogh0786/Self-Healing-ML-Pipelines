"""
Mathematical Data Drift Detector using Kullback-Leibler (KL) Divergence and Population Stability Index (PSI).
"""
import os
import json
import sqlite3
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from scipy.stats import entropy

from src.model.train_baseline import FEATURE_NAMES


def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes KL Divergence D_KL(P || Q) = sum_x P(x) * ln( P(x) / Q(x) ).
    p and q must be valid probability distributions summing to 1.
    """
    p_norm = p / np.sum(p)
    q_norm = q / np.sum(q)
    return float(np.sum(p_norm * np.log(p_norm / q_norm)))


def calculate_psi(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes Population Stability Index (PSI) = sum_i (Q_i - P_i) * ln( Q_i / P_i ).
    """
    p_norm = p / np.sum(p)
    q_norm = q / np.sum(q)
    return float(np.sum((q_norm - p_norm) * np.log(q_norm / p_norm)))


class DriftDetector:
    """
    Detects statistical data drift by comparing recent production request feature distributions Q(x)
    against the training baseline reference distributions P(x).
    """
    def __init__(
        self,
        baseline_path: str = "baseline_distribution.json",
        kl_threshold: float = 0.15,
        psi_threshold: float = 0.15,
        db_path: str = "logs.db",
        window_size: int = 1000,
        laplace_smoothing: float = 1e-4
    ):
        self.baseline_path = baseline_path
        self.kl_threshold = kl_threshold
        self.psi_threshold = psi_threshold
        self.db_path = db_path
        self.window_size = window_size
        self.laplace_smoothing = laplace_smoothing
        self.baseline_distributions = self._load_baseline()

    def _load_baseline(self) -> Dict[str, Any]:
        if not os.path.exists(self.baseline_path):
            raise FileNotFoundError(f"Baseline distributions file not found at {self.baseline_path}")
        with open(self.baseline_path, "r") as f:
            return json.load(f)

    def fetch_recent_logs(self) -> pd.DataFrame:
        """Fetches the last `window_size` inference logs from SQLite."""
        if not os.path.exists(self.db_path):
            return pd.DataFrame()
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT {', '.join(FEATURE_NAMES)} FROM inference_logs ORDER BY id DESC LIMIT {self.window_size}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def compute_feature_drift(self, col: str, live_values: np.ndarray) -> Dict[str, float]:
        """
        Computes D_KL and PSI for a single feature using the baseline's fixed bin edges.
        """
        baseline_info = self.baseline_distributions[col]
        bin_edges = np.array(baseline_info["bin_edges"])
        p_probs = np.array(baseline_info["probabilities"])

        # Histogram of live data using baseline bin edges
        counts, _ = np.histogram(live_values, bins=bin_edges)
        smoothed_counts = counts.astype(float) + self.laplace_smoothing
        q_probs = smoothed_counts / np.sum(smoothed_counts)

        kl_div = calculate_kl_divergence(p_probs, q_probs)
        psi = calculate_psi(p_probs, q_probs)

        return {
            "kl_divergence": kl_div,
            "psi": psi,
            "drift_detected": bool(kl_div > self.kl_threshold or psi > self.psi_threshold)
        }

    def detect_drift(self, df: Optional[pd.DataFrame] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluates drift across all features.
        
        Args:
            df: Optional dataframe of incoming features. If None, queries from logs.db.
            
        Returns:
            (drift_detected: bool, report_dict: Dict[str, Any])
        """
        if df is None:
            df = self.fetch_recent_logs()

        if df.empty or len(df) < 10:
            # Not enough logged requests to compute statistical divergence
            return False, {
                "drift_detected": False,
                "reason": f"Insufficient sample size ({len(df)} requests logged; minimum 10 required).",
                "feature_metrics": {}
            }

        feature_metrics = {}
        drifted_features: List[str] = []
        max_kl = 0.0

        for col in FEATURE_NAMES:
            if col in df.columns and col in self.baseline_distributions:
                metrics = self.compute_feature_drift(col, df[col].values)
                feature_metrics[col] = metrics
                max_kl = max(max_kl, metrics["kl_divergence"])
                if metrics["drift_detected"]:
                    drifted_features.append(col)

        overall_drift_detected = len(drifted_features) > 0

        mean_psi = float(sum(m["psi"] for m in feature_metrics.values()) / max(1, len(feature_metrics)))
        max_psi = float(max((m["psi"] for m in feature_metrics.values()), default=0.0))

        report = {
            "drift_detected": overall_drift_detected,
            "max_kl_divergence": float(max_kl),
            "max_psi": max_psi,
            "mean_psi": mean_psi,
            "kl_threshold": self.kl_threshold,
            "drifted_features": drifted_features,
            "num_samples_evaluated": len(df),
            "feature_metrics": feature_metrics
        }

        return overall_drift_detected, report
