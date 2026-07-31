"""
Unit tests for Airflow BranchPythonOperator decision logic.
"""
from unittest.mock import MagicMock
import pytest
from airflow_pipeline.dags.self_healing_mlops_dag import _branch_on_drift, _branch_on_promotion


def test_branch_on_drift_detected():
    """When healing decision is RETRAIN, branch operator must return 'trigger_retraining_task'."""
    mock_ti = MagicMock()
    mock_ti.xcom_pull.return_value = {"decision": {"action": "RETRAIN"}}
    
    next_task = _branch_on_drift(ti=mock_ti)
    assert next_task == "trigger_retraining_task"


def test_branch_on_no_drift():
    """When healing decision is NONE, branch operator must return 'no_drift_end'."""
    mock_ti = MagicMock()
    mock_ti.xcom_pull.return_value = {"decision": {"action": "NONE"}}
    
    next_task = _branch_on_drift(ti=mock_ti)
    assert next_task == "no_drift_end"


def test_branch_on_rollback():
    """When healing decision is ROLLBACK, branch operator must route to trigger_rollback_task."""
    mock_ti = MagicMock()
    mock_ti.xcom_pull.return_value = {"decision": {"action": "ROLLBACK"}}
    
    next_task = _branch_on_drift(ti=mock_ti)
    assert next_task == "trigger_rollback_task"


def test_branch_on_fallback():
    """When healing decision is FALLBACK, branch operator must route to trigger_fallback_task."""
    mock_ti = MagicMock()
    mock_ti.xcom_pull.return_value = {"decision": {"action": "FALLBACK"}}
    
    next_task = _branch_on_drift(ti=mock_ti)
    assert next_task == "trigger_fallback_task"


def test_branch_on_promotion_true():
    """When model is promoted, branch operator must route to K8s zero-downtime rollout."""
    mock_ti = MagicMock()
    mock_ti.xcom_pull.return_value = True
    
    next_task = _branch_on_promotion(ti=mock_ti)
    assert next_task == "kubernetes_zero_downtime_rollout_task"


def test_branch_on_promotion_false():
    """When candidate is not promoted, branch operator must route to keep existing model."""
    mock_ti = MagicMock()
    mock_ti.xcom_pull.return_value = False
    
    next_task = _branch_on_promotion(ti=mock_ti)
    assert next_task == "keep_existing_model_task"
