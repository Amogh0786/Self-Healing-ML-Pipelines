"""Deployment automation and Kubernetes zero-downtime rollout modules."""
from .k8s_rollout import execute_rolling_update, simulate_k8s_rollout

__all__ = ["execute_rolling_update", "simulate_k8s_rollout"]
