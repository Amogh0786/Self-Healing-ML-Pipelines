"""
End-to-End Self-Healing Automated Production MLOps Pipeline Demo.
Walks through the entire 6-phase closed-loop architecture:
1. Train Baseline & Serialize Feature Distributions P(x)
2. Normal Traffic & Verify Drift == False
3. Inject Data Drift & Quantify KL Divergence (Drift == True)
4. Automated Retraining (Airflow Branching)
5. MLflow Model Evaluation & Promotion
6. Kubernetes Zero-Downtime Rolling Deployment
"""
import os
import time
import shutil
from src.model.train_baseline import train_baseline_model
from simulator.traffic_generator import generate_traffic
from src.drift.detector import DriftDetector
from src.drift.report import generate_drift_report, save_drift_report
from src.model.retrain import run_retraining_pipeline
from src.model.evaluate import evaluate_model
from src.deploy.k8s_rollout import simulate_k8s_rollout
from src.healing.decision_engine import HybridDecisionEngine, generate_audit_summary


def print_header(title: str) -> None:
    print("\n" + "#" * 80)
    print(f"### {title.center(72)} ###")
    print("#" * 80 + "\n")


def run_demo() -> None:
    # Cleanup old logs for clean demo run
    if os.path.exists("logs.db"):
        os.remove("logs.db")
        
    print_header("PHASE 1: ESTABLISHING BASELINE MODEL & DISTRIBUTIONS")
    print("Training baseline Random Forest Regressor on California Housing dataset...")
    baseline_model, baseline_metrics, baseline_run_id = train_baseline_model(
        model_output_path="production_model.joblib",
        distribution_output_path="baseline_distribution.json"
    )
    print(f"-> Baseline Model Registered in MLflow [Run ID: {baseline_run_id}]")
    print(f"-> Baseline R2 Score: {baseline_metrics['r2_score']:.4f}")

    time.sleep(1)

    print_header("PHASE 2: SIMULATING NORMAL INCOMING TRAFFIC & VERIFYING NO DRIFT")
    print("Simulating 400 normal API requests with baseline feature distributions...")
    generate_traffic(num_requests=400, drift=False, api_url=None, db_path="logs.db")
    
    print("\nRunning daily drift evaluation via KL Divergence...")
    detector = DriftDetector(baseline_path="baseline_distribution.json", db_path="logs.db")
    drift_detected_normal, report_normal = detector.detect_drift()
    
    print(generate_drift_report(report_normal))
    assert not drift_detected_normal, "Drift should NOT be detected under normal traffic!"
    print("-> SUCCESS: KL Divergence remained below threshold (0.25). Pipeline ends peacefully.")

    time.sleep(1)

    print_header("PHASE 3: INJECTING SKEWED PRODUCTION DATA (THE TRIGGER)")
    print("Simulating 300 skewed API requests (3.5x median income, shifted age distribution)...")
    generate_traffic(num_requests=300, drift=True, api_url=None, db_path="logs.db")
    
    print("\nRe-evaluating statistical data drift across recent request window...")
    drift_detected_skewed, report_skewed = detector.detect_drift()
    save_drift_report(report_skewed, "drift_report.json")
    
    print(generate_drift_report(report_skewed))
    assert drift_detected_skewed, "Drift MUST be detected under skewed traffic!"
    print(f"-> ALERT TRIGGERED! Max KL Divergence ({report_skewed['max_kl_divergence']:.4f}) exceeded threshold (0.15).")
    
    print("\nEvaluating recovery strategy via Hybrid Decision Engine (Rules + Contextual Bandits)...")
    healing_engine = HybridDecisionEngine(drift_threshold=0.15)
    decision = healing_engine.decide_healing_action(
        max_kl=report_skewed.get("max_kl_divergence", 0.0),
        max_psi=report_skewed.get("max_psi", 0.0),
        sample_count=report_skewed.get("num_samples_evaluated", 700),
        error_rate=0.0,
        ignore_cooldown=True
    )
    print(generate_audit_summary(decision))
    assert decision["decision"]["action"] in ("RETRAIN", "ROLLBACK", "FALLBACK"), "Healing action must be selected!"
    print(f"-> Airflow BranchPythonOperator routing to: 'trigger_{decision['decision']['action'].lower()}_task'")

    time.sleep(1)

    print_header("PHASE 4: AUTOMATED RETRAINING ON UPDATED PRODUCTION DATA")
    print("Orchestrator spinning up retraining job with recent logs + baseline data...")
    candidate_model, candidate_metrics, candidate_run_id = run_retraining_pipeline(
        db_path="logs.db",
        n_estimators=100,
        max_depth=12,
        update_baseline_distribution=True
    )
    print(f"-> Candidate Model Registered in MLflow under 'Staging' [Run ID: {candidate_run_id}]")
    print(f"-> Candidate R2 Score: {candidate_metrics['r2_score']:.4f}")

    time.sleep(1)

    print_header("PHASE 5: MLFLOW VALIDATION & AUTOMATED PROMOTION")
    print("Comparing Candidate ('Staging') against current 'Production' baseline...")
    promoted, reason = evaluate_model(
        candidate_run_id=candidate_run_id,
        candidate_model_obj=candidate_model,
        metric_name="r2_score",
        higher_is_better=True
    )
    print(f"-> Promotion Status: {promoted}")
    print(f"-> Decision Rationale: {reason}")

    time.sleep(1)

    print_header("PHASE 6: KUBERNETES ZERO-DOWNTIME ROLLING DEPLOYMENT")
    if promoted:
        print("Model promoted in MLflow! Triggering Kubernetes RollingUpdate for serving pods...")
        simulate_k8s_rollout("fastapi-model-serving")
    else:
        print("Candidate model did not outperform production. No deployment update triggered.")

    print_header("CLOSED-LOOP SELF-HEALING MLOPS PIPELINE VERIFICATION COMPLETE")
    print("The pipeline successfully detected statistical drift, retrained, validated, and deployed!")


if __name__ == "__main__":
    run_demo()
