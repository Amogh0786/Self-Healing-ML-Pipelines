"""
Evaluation and automated model promotion module.
Compares a newly trained Staging candidate against the current Production model in MLflow.
"""
from typing import Any, Tuple, Dict
from src.mlflow_utils.model_manager import ModelManager


def evaluate_model(
    candidate_run_id: str,
    candidate_model_obj: Any,
    metric_name: str = "r2_score",
    higher_is_better: bool = True
) -> Tuple[bool, str]:
    """
    Evaluates candidate model run against current Production model.
    If the candidate outperforms Production, it is promoted to Production.
    
    Args:
        candidate_run_id: MLflow Run ID of the candidate Staging model.
        candidate_model_obj: The in-memory scikit-learn model object.
        metric_name: Name of the evaluation metric to compare (default: 'r2_score').
        higher_is_better: Whether higher values indicate better performance.
        
    Returns:
        (promoted: bool, reason: str)
    """
    manager = ModelManager()
    promoted, reason = manager.compare_and_promote(
        candidate_run_id=candidate_run_id,
        candidate_model_obj=candidate_model_obj,
        metric_name=metric_name,
        higher_is_better=higher_is_better
    )
    print(f"[Model Evaluation] {reason}")
    return promoted, reason
