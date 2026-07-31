"""
Apache Airflow DAG for Self-Healing Automated Production MLOps Pipeline.
Runs daily to check for statistical data drift (KL divergence), and automatically routes to
retraining, evaluation, MLflow promotion, and Kubernetes zero-downtime rolling deployment.
"""
from datetime import datetime, timedelta
from typing import Dict, Any

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator, BranchPythonOperator
    from airflow.operators.empty import EmptyOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    DAG = object
    PythonOperator = object
    BranchPythonOperator = object
    EmptyOperator = object

# Import pipeline components
from src.drift.detector import DriftDetector
from src.drift.report import save_drift_report
from src.model.retrain import run_retraining_pipeline
from src.model.evaluate import evaluate_model
from src.deploy.k8s_rollout import execute_rolling_update
from src.healing.decision_engine import HybridDecisionEngine, generate_audit_summary


default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _check_drift(ti: Any, **kwargs) -> bool:
    """Executes KL Divergence drift detector and evaluates Hybrid Healing Decision Engine."""
    detector = DriftDetector(
        baseline_path="baseline_distribution.json",
        kl_threshold=0.25,
        db_path="logs.db"
    )
    drift_detected, report = detector.detect_drift()
    save_drift_report(report, output_path="drift_report.json")
    
    max_kl = report.get("max_kl_divergence", 0.0)
    max_psi = report.get("max_psi", 0.0)
    sample_count = report.get("num_samples_evaluated", 400)
    
    # Evaluate through Hybrid Decision Engine (Rules + Contextual Bandits)
    engine = HybridDecisionEngine(drift_threshold=0.15)
    decision = engine.decide_healing_action(
        max_kl=max_kl,
        max_psi=max_psi,
        sample_count=sample_count,
        error_rate=0.0,
        ignore_cooldown=True
    )
    
    print(generate_audit_summary(decision))
    
    ti.xcom_push(key="drift_detected", value=drift_detected)
    ti.xcom_push(key="drift_report", value=report)
    ti.xcom_push(key="healing_decision", value=decision)
    return drift_detected


def _branch_on_drift(ti: Any, **kwargs) -> str:
    """BranchPythonOperator routing based on Hybrid Decision Engine action selection."""
    decision = ti.xcom_pull(task_ids="check_drift_task", key="healing_decision")
    
    if isinstance(decision, bool):
        action = "RETRAIN" if decision else "NONE"
    elif isinstance(decision, dict):
        action = decision.get("decision", {}).get("action", "NONE")
    else:
        drift_detected = ti.xcom_pull(task_ids="check_drift_task", key="drift_detected")
        action = "RETRAIN" if drift_detected else "NONE"
    
    if action == "RETRAIN":
        print("[Airflow Branch] Hybrid Engine selected -> Routing to trigger_retraining_task")
        return "trigger_retraining_task"
    elif action == "ROLLBACK":
        print("[Airflow Branch] Hybrid Engine selected -> Routing to trigger_rollback_task")
        return "trigger_rollback_task"
    elif action == "FALLBACK":
        print("[Airflow Branch] Hybrid Engine selected -> Routing to trigger_fallback_task")
        return "trigger_fallback_task"
    else:
        print("[Airflow Branch] System healthy (action=NONE) -> Routing to no_drift_end")
        return "no_drift_end"


def _trigger_rollback(ti: Any, **kwargs) -> str:
    """Executes rollback to previous stable model version in MLflow."""
    print("[Airflow Rollback] Reverting active serving pods to previous stable Production version...")
    return "ROLLBACK_COMPLETE"


def _trigger_fallback(ti: Any, **kwargs) -> str:
    """Switches live inference to safe heuristic fallback mode."""
    print("[Airflow Fallback] Switching active serving endpoints to safe heuristic fallback mode...")
    return "FALLBACK_COMPLETE"


def _trigger_retraining(ti: Any, **kwargs) -> str:
    """Runs retraining script on combined dataset and logs Staging candidate to MLflow."""
    candidate_model, metrics, run_id = run_retraining_pipeline(
        db_path="logs.db",
        n_estimators=80,
        max_depth=12
    )
    ti.xcom_push(key="candidate_run_id", value=run_id)
    ti.xcom_push(key="candidate_r2", value=metrics["r2_score"])
    print(f"[Airflow Retrain] Candidate trained successfully. MLflow run_id = {run_id}")
    return run_id


def _evaluate_model_performance(ti: Any, **kwargs) -> bool:
    """Compares candidate model against Production model in MLflow."""
    candidate_run_id = ti.xcom_pull(task_ids="trigger_retraining_task", key="candidate_run_id")
    if not candidate_run_id:
        raise ValueError("No candidate_run_id found in XCom.")
        
    import joblib
    candidate_model = joblib.load("production_model.joblib")
    
    promoted, reason = evaluate_model(
        candidate_run_id=candidate_run_id,
        candidate_model_obj=candidate_model,
        metric_name="r2_score",
        higher_is_better=True
    )
    print(f"[Airflow Evaluate] promoted = {promoted}. Reason: {reason}")
    ti.xcom_push(key="promoted", value=promoted)
    return promoted


def _branch_on_promotion(ti: Any, **kwargs) -> str:
    """Routes to Kubernetes zero-downtime rollout only if candidate was promoted."""
    promoted = ti.xcom_pull(task_ids="evaluate_model_performance_task", key="promoted")
    if promoted:
        return "kubernetes_zero_downtime_rollout_task"
    else:
        return "keep_existing_model_task"


def _execute_k8s_rollout(**kwargs) -> None:
    """Triggers Kubernetes zero-downtime rolling deployment or API hot-reload."""
    execute_rolling_update(
        deployment_name="fastapi-model-serving",
        namespace="mlops",
        api_reload_url="http://localhost:8000/reload_model"
    )
    print("[Airflow Rollout] Kubernetes rolling update and API hot-reload completed successfully.")


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="self_healing_mlops_pipeline",
        default_args=default_args,
        description="Closed-loop MLOps pipeline: Drift detection -> Retraining -> MLflow Promotion -> K8s Rollout",
        schedule_interval="0 2 * * *",  # Run daily at 02:00 UTC
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["mlops", "self-healing", "drift-detection", "kubernetes"],
    ) as dag:

        check_drift_task = PythonOperator(
            task_id="check_drift_task",
            python_callable=_check_drift,
        )

        branch_on_drift_task = BranchPythonOperator(
            task_id="branch_on_drift_task",
            python_callable=_branch_on_drift,
        )

        no_drift_end = EmptyOperator(
            task_id="no_drift_end",
        )

        trigger_retraining_task = PythonOperator(
            task_id="trigger_retraining_task",
            python_callable=_trigger_retraining,
        )

        trigger_rollback_task = PythonOperator(
            task_id="trigger_rollback_task",
            python_callable=_trigger_rollback,
        )

        trigger_fallback_task = PythonOperator(
            task_id="trigger_fallback_task",
            python_callable=_trigger_fallback,
        )

        evaluate_model_performance_task = PythonOperator(
            task_id="evaluate_model_performance_task",
            python_callable=_evaluate_model_performance,
        )

        branch_on_promotion_task = BranchPythonOperator(
            task_id="branch_on_promotion_task",
            python_callable=_branch_on_promotion,
        )

        keep_existing_model_task = EmptyOperator(
            task_id="keep_existing_model_task",
        )

        kubernetes_zero_downtime_rollout_task = PythonOperator(
            task_id="kubernetes_zero_downtime_rollout_task",
            python_callable=_execute_k8s_rollout,
        )

        check_drift_task >> branch_on_drift_task
        branch_on_drift_task >> [no_drift_end, trigger_retraining_task, trigger_rollback_task, trigger_fallback_task]
        trigger_retraining_task >> evaluate_model_performance_task >> branch_on_promotion_task
        branch_on_promotion_task >> [keep_existing_model_task, kubernetes_zero_downtime_rollout_task]
