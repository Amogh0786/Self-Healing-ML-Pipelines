"""
Baseline model training on California Housing dataset and feature distribution serialization.
"""
import os
import json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from src.mlflow_utils.tracking import setup_mlflow, log_model_with_metrics, compute_dataset_hash
from src.mlflow_utils.model_manager import ModelManager


FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]


def compute_feature_histogram(
    values: np.ndarray,
    num_bins: int = 20,
    laplace_smoothing: float = 1e-4
) -> Dict[str, Any]:
    """
    Computes quantile/equal-width bin edges and Laplace-smoothed probability distribution P(x)
    for a continuous feature.
    """
    # Use equal-width binning between min and max
    min_val, max_val = float(np.min(values)), float(np.max(values))
    if min_val == max_val:
        max_val += 1e-6
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    counts, _ = np.histogram(values, bins=bin_edges)
    # Apply Laplace smoothing so no bin has probability 0 (crucial for KL divergence ln(P/Q))
    smoothed_counts = counts.astype(float) + laplace_smoothing
    probabilities = smoothed_counts / np.sum(smoothed_counts)
    
    return {
        "bin_edges": bin_edges.tolist(),
        "probabilities": probabilities.tolist(),
        "min": min_val,
        "max": max_val,
        "mean": float(np.mean(values)),
        "std": float(np.std(values))
    }


def save_baseline_distributions(
    df: pd.DataFrame,
    output_path: str = "baseline_distribution.json",
    num_bins: int = 10
) -> Dict[str, Any]:
    """
    Computes and saves the baseline distribution P(x) for each feature in the training dataset.
    """
    distributions = {}
    for col in FEATURE_NAMES:
        if col in df.columns:
            distributions[col] = compute_feature_histogram(df[col].values, num_bins=num_bins)
            
    with open(output_path, "w") as f:
        json.dump(distributions, f, indent=2)
        
    return distributions


def train_baseline_model(
    model_output_path: str = "production_model.joblib",
    distribution_output_path: str = "baseline_distribution.json",
    n_estimators: int = 50,
    max_depth: int = 10,
    random_state: int = 42
) -> Tuple[Any, Dict[str, float], str]:
    """
    Trains baseline Random Forest model, serializes reference feature distribution P(x),
    and logs the model to MLflow as the initial Production model.
    """
    print("Loading California Housing dataset...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df[FEATURE_NAMES]
    y = df["MedHouseVal"]
    
    # Save baseline distributions P(x) for KL divergence comparison
    save_baseline_distributions(X, output_path=distribution_output_path)
    print(f"Saved baseline feature distributions to {distribution_output_path}")
    
    # Compute dataset hash
    dataset_hash = compute_dataset_hash(X.values)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    print(f"Training RandomForestRegressor(n_estimators={n_estimators}, max_depth={max_depth})...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = float(root_mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2
    }
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": random_state
    }
    
    print(f"Baseline Evaluation -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    # Setup MLflow and log model
    setup_mlflow(experiment_name="california_housing_drift")
    run_id = log_model_with_metrics(
        model=model,
        model_name="CaliforniaHousingRegressor",
        metrics=metrics,
        params=params,
        dataset_hash=dataset_hash,
        stage_tag="Production"
    )
    
    # Save locally for FastAPI serving
    joblib.dump(model, model_output_path)
    joblib.dump(model, "baseline_model.joblib")
    print(f"Saved baseline model to {model_output_path}")
    
    return model, metrics, run_id


if __name__ == "__main__":
    train_baseline_model()
