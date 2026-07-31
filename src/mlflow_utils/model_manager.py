"""
Model Registry management and automated promotion logic.
"""
import os
import shutil
from typing import Optional, Dict, Any, Tuple
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


class ModelManager:
    """
    Manages model versions, lifecycle tags/aliases, and automated promotion in MLflow.
    Also syncs the currently promoted production model to a local artifact path for rapid FastAPI loading.
    """
    def __init__(
        self,
        registered_model_name: str = "CaliforniaHousingRegressor",
        local_model_path: str = "production_model.joblib"
    ):
        self.model_name = registered_model_name
        self.local_model_path = local_model_path
        self.client = MlflowClient()

    def get_latest_version(self, stage_tag: str = "Production") -> Optional[Any]:
        """
        Retrieves the latest model version matching the stage tag or alias.
        """
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            # Sort by version number descending
            sorted_versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
            for v in sorted_versions:
                run = self.client.get_run(v.run_id)
                if run.data.tags.get("model_stage") == stage_tag:
                    return v
            return None
        except MlflowException:
            return None

    def get_model_metrics(self, run_id: str) -> Dict[str, float]:
        """Fetches logged metrics for a given run ID."""
        run = self.client.get_run(run_id)
        return {k: float(v) for k, v in run.data.metrics.items()}

    def compare_and_promote(
        self,
        candidate_run_id: str,
        candidate_model_obj: Any,
        metric_name: str = "r2_score",
        higher_is_better: bool = True
    ) -> Tuple[bool, str]:
        """
        Compares the candidate run against the current production model.
        Promotes if candidate outperforms or if no production model exists.
        
        Returns:
            (promoted: bool, reason: str)
        """
        candidate_metrics = self.get_model_metrics(candidate_run_id)
        candidate_score = candidate_metrics.get(metric_name, 0.0)

        prod_version = self.get_latest_version(stage_tag="Production")
        
        if prod_version is None:
            # No production model exists yet; promote immediately
            self.promote_run_to_production(candidate_run_id, candidate_model_obj)
            return True, f"No existing Production model found. Promoted candidate with {metric_name}={candidate_score:.4f}."

        prod_metrics = self.get_model_metrics(prod_version.run_id)
        prod_score = prod_metrics.get(metric_name, 0.0)

        is_better = (
            (candidate_score > prod_score) if higher_is_better 
            else (candidate_score < prod_score)
        )

        if is_better:
            # Demote old production run tag
            try:
                self.client.set_tag(prod_version.run_id, "model_stage", "Archived")
            except Exception:
                pass
                
            self.promote_run_to_production(candidate_run_id, candidate_model_obj)
            return True, (
                f"Candidate model outperformed Production ({metric_name}: {candidate_score:.4f} vs {prod_score:.4f}). "
                f"Promoted to Production."
            )
        else:
            return False, (
                f"Candidate model did not outperform Production ({metric_name}: {candidate_score:.4f} vs {prod_score:.4f}). "
                f"Kept existing Production model."
            )

    def promote_run_to_production(self, run_id: str, model_obj: Any) -> None:
        """Sets MLflow stage tag to Production and saves model locally for API serving."""
        try:
            self.client.set_tag(run_id, "model_stage", "Production")
        except Exception:
            pass
            
        # Save locally for FastAPI hot reload
        joblib.dump(model_obj, self.local_model_path)
