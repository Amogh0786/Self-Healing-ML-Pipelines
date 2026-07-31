from typing import Tuple, Dict, Any, List
import math


class ContextualBanditPolicy:
    """
    Contextual Bandit / Bayesian Expected Utility Action Selector.

    Evaluates context vector x = [max_kl, mean_psi, sample_count, error_rate] and scores
    healing arms: RETRAIN, ROLLBACK, FALLBACK.
    """

    def __init__(
        self,
        drift_threshold: float = 0.25,
        min_retrain_samples: int = 50,
        high_error_threshold: float = 0.15
    ):
        self.drift_threshold = drift_threshold
        self.min_retrain_samples = min_retrain_samples
        self.high_error_threshold = high_error_threshold

    def select_action(self, context: Dict[str, Any]) -> Tuple[str, float, str, Dict[str, float]]:
        """
        Selects the optimal healing action based on context utility scores.

        Args:
            context: Dictionary containing 'max_kl', 'mean_psi', 'sample_count', 'error_rate'.

        Returns:
            Tuple[str, float, str, Dict[str, float]]:
                (selected_action, confidence_score, rationale, utility_scores)
        """
        max_kl = float(context.get("max_kl", 0.0))
        mean_psi = float(context.get("mean_psi", 0.0))
        sample_count = int(context.get("sample_count", 0))
        error_rate = float(context.get("error_rate", 0.0))

        # Calculate posterior expected utility for each arm
        utilities: Dict[str, float] = {}

        # 1. RETRAIN Utility: Scales with KL divergence and sample statistical significance
        if sample_count >= self.min_retrain_samples and max_kl >= self.drift_threshold:
            sample_factor = min(1.0, sample_count / 300.0)
            drift_factor = min(1.0, max_kl / 0.60)
            utilities["RETRAIN"] = 0.82 + (0.15 * sample_factor * drift_factor)
        else:
            utilities["RETRAIN"] = 0.30

        # 2. ROLLBACK Utility: High when error rate spikes rapidly on a recent deployment
        if error_rate >= self.high_error_threshold:
            utilities["ROLLBACK"] = 0.85 + min(0.10, error_rate * 0.5)
        else:
            utilities["ROLLBACK"] = 0.20

        # 3. FALLBACK Utility: High when sample size is too small to safely retrain
        if sample_count < self.min_retrain_samples and max_kl >= self.drift_threshold:
            utilities["FALLBACK"] = 0.88
        else:
            utilities["FALLBACK"] = 0.15

        # Select arm with maximum utility
        best_action = max(utilities.items(), key=lambda x: x[1])
        action, raw_score = best_action
        confidence = min(0.99, raw_score)

        # Generate human-readable rationale
        if action == "RETRAIN":
            rationale = (
                f"Contextual Bandit Policy selected RETRAIN (Utility: {confidence:.2f}): "
                f"Statistically significant drift (max_kl={max_kl:.4f} >= {self.drift_threshold}) "
                f"with robust sample size ({sample_count} >= {self.min_retrain_samples})."
            )
        elif action == "ROLLBACK":
            rationale = (
                f"Contextual Bandit Policy selected ROLLBACK (Utility: {confidence:.2f}): "
                f"High prediction error rate detected (error_rate={error_rate:.2f} >= {self.high_error_threshold})."
            )
        elif action == "FALLBACK":
            rationale = (
                f"Contextual Bandit Policy selected FALLBACK (Utility: {confidence:.2f}): "
                f"Drift detected but sample size ({sample_count}) is below minimum retraining threshold ({self.min_retrain_samples})."
            )
        else:
            rationale = "No action selected."

        return action, confidence, rationale, utilities
