import os
import json
import pytest
import time
from src.healing.rules_engine import RulesEngine
from src.healing.bandit_policy import ContextualBanditPolicy
from src.healing.decision_engine import HybridDecisionEngine, generate_audit_summary


@pytest.fixture
def temp_audit_log(tmp_path):
    return str(tmp_path / "test_healing_audit_log.json")


def test_rules_engine_cooldown(temp_audit_log):
    """Verifies that cooldown timer blocks rapid sequential healing actions."""
    rules = RulesEngine(cooldown_seconds=1800, audit_log_path=temp_audit_log)
    
    # No history -> no cooldown
    in_cooldown, reason = rules.check_cooldown()
    assert not in_cooldown
    
    # Write fake entry with recent timestamp
    entry = [{
        "timestamp_unix": time.time(),
        "decision": {"action": "RETRAIN"}
    }]
    with open(temp_audit_log, "w", encoding="utf-8") as f:
        json.dump(entry, f)
        
    in_cooldown, reason = rules.check_cooldown()
    assert in_cooldown
    assert "Cooldown period active" in reason


def test_rules_engine_critical_override(temp_audit_log):
    """Verifies that severe drift (max_kl >= 0.80) triggers immediate RETRAIN override."""
    rules = RulesEngine(critical_kl_threshold=0.80, audit_log_path=temp_audit_log)
    override, action, rationale = rules.evaluate_override(max_kl=0.85, max_psi=0.4, ignore_cooldown=True)
    assert override
    assert action == "RETRAIN"
    assert "Critical Data Drift Override" in rationale


def test_bandit_policy_arm_selection():
    """Verifies Contextual Bandit expected utility scoring for RETRAIN, ROLLBACK, FALLBACK."""
    bandit = ContextualBanditPolicy(drift_threshold=0.25, min_retrain_samples=50, high_error_threshold=0.15)
    
    # 1. Statistically significant drift -> RETRAIN
    action1, conf1, rat1, util1 = bandit.select_action({
        "max_kl": 0.35,
        "mean_psi": 0.20,
        "sample_count": 400,
        "error_rate": 0.05
    })
    assert action1 == "RETRAIN"
    assert conf1 >= 0.80
    
    # 2. High prediction error rate spike -> ROLLBACK
    action2, conf2, rat2, util2 = bandit.select_action({
        "max_kl": 0.10,
        "mean_psi": 0.05,
        "sample_count": 200,
        "error_rate": 0.25
    })
    assert action2 == "ROLLBACK"
    
    # 3. Drift detected but sample size too small -> FALLBACK
    action3, conf3, rat3, util3 = bandit.select_action({
        "max_kl": 0.30,
        "mean_psi": 0.15,
        "sample_count": 20,
        "error_rate": 0.0
    })
    assert action3 == "FALLBACK"


def test_hybrid_decision_engine_end_to_end(temp_audit_log):
    """Verifies full Hybrid Decision Engine control loop and structured JSON audit logging."""
    engine = HybridDecisionEngine(audit_log_path=temp_audit_log, drift_threshold=0.25)
    
    # Healthy state
    res_healthy = engine.decide_healing_action(max_kl=0.10, mean_psi=0.05, sample_count=300)
    assert res_healthy["decision"]["action"] == "NONE"
    
    # Drifted state
    res_drift = engine.decide_healing_action(max_kl=0.35, mean_psi=0.40, sample_count=350, ignore_cooldown=True)
    assert res_drift["decision"]["action"] == "RETRAIN"
    assert res_drift["decision"]["source"] == "CONTEXTUAL_BANDIT"
    
    # Verify audit summary formatter
    summary_str = generate_audit_summary(res_drift)
    assert "Selected Action   : RETRAIN" in summary_str
    
    # Verify audit log JSON file was written
    assert os.path.exists(temp_audit_log)
    with open(temp_audit_log, "r", encoding="utf-8") as f:
        logs = json.load(f)
        assert len(logs) == 2
        assert logs[1]["decision"]["action"] == "RETRAIN"
