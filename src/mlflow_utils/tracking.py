"""
MLflow tracking utilities for experiment lifecycle and autologging.
"""
import os
import hashlib
from typing import Dict, Any, Optional
import mlflow
import mlflow.sklearn
import numpy as np


def setup_mlflow(
    experiment_name: str = "california_housing_self_healing",
    tracking_uri: Optional[str] = None
) -> str:
    """
    Configures MLflow tracking URI and experiment name.
    
    Args:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: URI for tracking server (defaults to MLFLOW_TRACKING_URI or local mlruns).
        
    Returns:
        experiment_id: Unique MLflow experiment ID.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    
    # Enable scikit-learn autologging
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True,
        silent=True
    )
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
        
    mlflow.set_experiment(experiment_name)
    return experiment_id


def compute_dataset_hash(data_array: np.ndarray) -> str:
    """Computes a SHA256 hash of a numpy array or dataframe to track data version."""
    data_bytes = np.ascontiguousarray(data_array).tobytes()
    return hashlib.sha256(data_bytes).hexdigest()[:12]


def log_model_with_metrics(
    model: Any,
    model_name: str,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    dataset_hash: str,
    stage_tag: str = "Staging",
    artifact_path: str = "model"
) -> str:
    """
    Logs hyperparameters, validation metrics, dataset version, and registers the model.
    
    Args:
        model: Trained scikit-learn model.
        model_name: Registered model name in MLflow.
        metrics: Dictionary of metric names and float values (e.g., {'r2': 0.82, 'rmse': 0.45}).
        params: Hyperparameters dictionary.
        dataset_hash: SHA256 substring identifying training data version.
        stage_tag: Tag indicating lifecycle stage ('Production' or 'Staging').
        artifact_path: MLflow artifact directory for model weights.
        
    Returns:
        run_id: MLflow Run ID for the logged model.
    """
    with mlflow.start_run() as run:
        # Log dataset hash & custom tags
        mlflow.set_tag("dataset_hash", dataset_hash)
        mlflow.set_tag("model_stage", stage_tag)
        
        # Log parameters
        for k, v in params.items():
            mlflow.log_param(k, v)
            
        # Log evaluation metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
            
        # Log model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=model_name
        )
        
        return run.info.run_id
