import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional
from src.healing.rules_engine import RulesEngine
from src.healing.bandit_policy import ContextualBanditPolicy


class HybridDecisionEngine:
    """
    Hybrid Control Engine for Safe Autonomous ML Operations.

    Combines:
      Layer 1: Deterministic Safety Rules (Cooldown, Critical Severity Override, Confidence Gating).
      Layer 2: Probabilistic Contextual Bandit Policy (Retrain vs. Rollback vs. Fallback).

    Produces structured JSON audit trails for full ISO 27001-ready observability.
    """

    def __init__(
        self,
        audit_log_path: str = "healing_audit_log.json",
        cooldown_seconds: int = 1800,
        drift_threshold: float = 0.15,
        psi_threshold: float = 0.15,
        critical_kl_threshold: float = 0.80,
        min_confidence: float = 0.80
    ):
        self.audit_log_path = audit_log_path
        self.drift_threshold = drift_threshold
        self.psi_threshold = psi_threshold
        self.rules_engine = RulesEngine(
            cooldown_seconds=cooldown_seconds,
            critical_kl_threshold=critical_kl_threshold,
            min_confidence=min_confidence,
            audit_log_path=audit_log_path
        )
        self.bandit_policy = ContextualBanditPolicy(
            drift_threshold=drift_threshold
        )

    def decide_healing_action(
        self,
        max_kl: float,
        max_psi: float = 0.0,
        sample_count: int = 400,
        error_rate: float = 0.0,
        ignore_cooldown: bool = False,
        mean_psi: float = 0.0,
        recent_mse: float = 0.0,
        baseline_mse: float = 1.0
    ) -> Dict[str, Any]:
        """
        Evaluates context through the hybrid control loop and returns an auditable decision payload.

        Args:
            max_kl: Maximum KL divergence observed across features.
            max_psi: Maximum PSI score observed across features.
            sample_count: Number of recent logged samples evaluated.
            error_rate: Observed prediction error rate or error spike metric.
            ignore_cooldown: Set True for testing/simulations to bypass cooldown timers.
            mean_psi: Alternative PSI metric argument for compatibility.
            recent_mse: Actual mean squared error based on delayed ground truth (Concept Drift).
            baseline_mse: The historical baseline MSE for comparison.

        Returns:
            Dict[str, Any]: Structured JSON decision payload containing action, confidence, and rationale.
        """
        psi_val = max(max_psi, mean_psi)
        context = {
            "max_kl": max_kl,
            "mean_psi": psi_val,
            "sample_count": sample_count,
            "error_rate": error_rate,
            "recent_mse": recent_mse,
            "baseline_mse": baseline_mse
        }

        # Step 1: Check if drift is below threshold -> No action needed
        mse_ratio = recent_mse / baseline_mse if baseline_mse > 0 else 1.0
        if max_kl < self.drift_threshold and psi_val < self.psi_threshold and error_rate < self.bandit_policy.high_error_threshold and mse_ratio < 1.25:
            return self._record_decision(
                context=context,
                action="NONE",
                confidence=1.0,
                source="DETERMINISTIC_RULE",
                rationale=f"Data drift (KL={max_kl:.4f}, PSI={psi_val:.4f}) and Concept Drift (MSE Ratio={mse_ratio:.2f}) are below thresholds. System healthy."
            )

        # Step 2: Check Deterministic Override Rules (Cooldown / Critical Severity)
        override_triggered, override_action, override_rationale = self.rules_engine.evaluate_override(
            max_kl=max_kl,
            max_psi=psi_val,
            ignore_cooldown=ignore_cooldown
        )
        if override_triggered:
            return self._record_decision(
                context=context,
                action=override_action,
                confidence=1.0,
                source="DETERMINISTIC_RULE",
                rationale=override_rationale
            )

        # Step 3: Evaluate Contextual Bandit / Bayesian Policy
        bandit_action, raw_confidence, bandit_rationale, utilities = self.bandit_policy.select_action(context)

        # Step 4: Apply Deterministic Confidence Gate
        final_action, final_confidence, gate_rationale = self.rules_engine.check_confidence_gate(
            confidence=raw_confidence,
            proposed_action=bandit_action
        )

        source = "CONTEXTUAL_BANDIT" if final_action != "NONE" else "CONFIDENCE_GATE_REJECTION"
        combined_rationale = f"{bandit_rationale} | {gate_rationale}"

        return self._record_decision(
            context=context,
            action=final_action,
            confidence=final_confidence,
            source=source,
            rationale=combined_rationale,
            utilities=utilities
        )

    def _record_decision(
        self,
        context: Dict[str, Any],
        action: str,
        confidence: float,
        source: str,
        rationale: str,
        utilities: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Records structured decision entry to JSON audit log and returns payload.
        """
        ts_unix = time.time()
        ts_iso = datetime.now(timezone.utc).isoformat()
        payload = {
            "timestamp_iso": ts_iso,
            "timestamp_unix": ts_unix,
            "context": context,
            "decision": {
                "action": action,
                "confidence": round(confidence, 4),
                "source": source
            },
            "rationale": rationale,
            "utilities": utilities or {}
        }

        # Append to audit log
        logs = []
        if os.path.exists(self.audit_log_path):
            try:
                with open(self.audit_log_path, "r", encoding="utf-8") as f:
                    logs = json.load(f)
                    if not isinstance(logs, list):
                        logs = []
            except Exception:
                logs = []
        logs.append(payload)
        try:
            with open(self.audit_log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write healing audit log: {e}")

        return payload


def generate_audit_summary(payload: Dict[str, Any]) -> str:
    """
    Formats an audit log entry as a clean CLI table/report.
    """
    decision = payload.get("decision", {})
    action = decision.get("action", "UNKNOWN")
    conf = decision.get("confidence", 0.0)
    source = decision.get("source", "UNKNOWN")
    rationale = payload.get("rationale", "")
    context = payload.get("context", {})

    lines = [
        "=" * 70,
        "          HYBRID HEALING DECISION AUDIT RECORD          ",
        "=" * 70,
        f"Selected Action   : {action}",
        f"Confidence Score  : {conf:.4f} (Source: {source})",
        f"Observed Max KL   : {context.get('max_kl', 0.0):.4f}",
        f"Sample Size       : {context.get('sample_count', 0)}",
        f"Decision Rationale: {rationale}",
        "=" * 70
    ]
    return "\n".join(lines)
