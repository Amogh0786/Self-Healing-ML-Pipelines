import os
import json
import time
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any


class RulesEngine:
    """
    Deterministic Safety Guardrails for Autonomous ML Healing.
    
    Enforces non-negotiable enterprise safety rules:
      1. Cooldown Enforcement: Prevents cascade healing loops (default 30-min window).
      2. Critical Severity Override: Automatically forces RETRAIN when KL Divergence >= 0.80.
      3. Confidence Gating: Rejects bandit actions below 80% statistical confidence.
    """

    def __init__(
        self,
        cooldown_seconds: int = 1800,
        critical_kl_threshold: float = 0.80,
        critical_psi_threshold: float = 1.0,
        min_confidence: float = 0.80,
        audit_log_path: str = "healing_audit_log.json"
    ):
        self.cooldown_seconds = cooldown_seconds
        self.critical_kl_threshold = critical_kl_threshold
        self.critical_psi_threshold = critical_psi_threshold
        self.min_confidence = min_confidence
        self.audit_log_path = audit_log_path

    def _get_last_healing_timestamp(self) -> Optional[float]:
        """Reads the last executed healing action timestamp from the audit log."""
        if not os.path.exists(self.audit_log_path):
            return None
        try:
            with open(self.audit_log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
                if not isinstance(logs, list) or len(logs) == 0:
                    return None
                for entry in reversed(logs):
                    action = entry.get("decision", {}).get("action")
                    if action in ("RETRAIN", "ROLLBACK", "FALLBACK"):
                        ts = entry.get("timestamp_unix", 0)
                        return float(ts)
        except Exception:
            return None
        return None

    def check_cooldown(self, current_time: Optional[float] = None) -> Tuple[bool, str]:
        """
        Checks if the system is currently within a cooldown period after a recent healing action.

        Returns:
            Tuple[bool, str]: (is_cooldown_active, reason)
        """
        if current_time is None:
            current_time = time.time()
        last_ts = self._get_last_healing_timestamp()
        if last_ts is not None and (current_time - last_ts) < self.cooldown_seconds:
            remaining_min = int((self.cooldown_seconds - (current_time - last_ts)) / 60)
            return True, f"Cooldown period active for another {remaining_min} mins"
        return False, "No active cooldown"

    def evaluate_override(
        self,
        max_kl: float,
        max_psi: float,
        ignore_cooldown: bool = False
    ) -> Tuple[bool, str, str]:
        """
        Evaluates deterministic override rules before consulting probabilistic policies.

        Returns:
            Tuple[bool, str, str]: (override_triggered, action, rationale)
        """
        if not ignore_cooldown:
            in_cooldown, reason = self.check_cooldown()
            if in_cooldown:
                return True, "NONE", f"Deterministic Block: {reason}"

        # Critical severity override
        if max_kl >= self.critical_kl_threshold or max_psi >= self.critical_psi_threshold:
            rationale = (
                f"Critical Data Drift Override: max_kl={max_kl:.4f} >= {self.critical_kl_threshold} "
                f"or max_psi={max_psi:.4f} >= {self.critical_psi_threshold}. Forcing immediate RETRAIN."
            )
            return True, "RETRAIN", rationale

        return False, "NONE", "No deterministic override triggered"

    def check_confidence_gate(self, confidence: float, proposed_action: str) -> Tuple[str, float, str]:
        """
        Enforces minimum statistical confidence threshold for autonomous actions.
        """
        if confidence < self.min_confidence:
            rationale = (
                f"Confidence Gate Rejection: Proposed action '{proposed_action}' confidence "
                f"({confidence:.2f}) < required threshold ({self.min_confidence:.2f}). Defaulting to NONE."
            )
            return "NONE", 1.0, rationale
        return proposed_action, confidence, f"Confidence check passed ({confidence:.2f} >= {self.min_confidence:.2f})"
