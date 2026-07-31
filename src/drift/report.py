"""
Drift report generation and formatting utilities.
"""
import json
from typing import Dict, Any


def generate_drift_report(report: Dict[str, Any]) -> str:
    """
    Formats the statistical drift detection report into a human-readable string/table.
    """
    lines = [
        "=" * 70,
        "                   STATISTICAL DATA DRIFT REPORT                  ",
        "=" * 70,
        f"Drift Detected : {report.get('drift_detected', False)}",
        f"Samples Tested : {report.get('num_samples_evaluated', 0)}",
        f"Max D_KL Score : {report.get('max_kl_divergence', 0.0):.4f} (Threshold: {report.get('kl_threshold', 0.25)})",
        f"Drifted Columns: {', '.join(report.get('drifted_features', [])) or 'None'}",
        "-" * 70,
        f"{'Feature':<15} | {'D_KL(P || Q)':<15} | {'PSI':<15} | {'Status':<10}",
        "-" * 70,
    ]
    
    feature_metrics = report.get("feature_metrics", {})
    for feature, metrics in feature_metrics.items():
        kl = metrics.get("kl_divergence", 0.0)
        psi = metrics.get("psi", 0.0)
        status = "DRIFT!" if metrics.get("drift_detected", False) else "OK"
        lines.append(f"{feature:<15} | {kl:<15.4f} | {psi:<15.4f} | {status:<10}")
        
    lines.append("=" * 70)
    return "\n".join(lines)


def save_drift_report(report: Dict[str, Any], output_path: str = "drift_report.json") -> None:
    """Saves the drift report dictionary to JSON."""
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved drift evaluation report to {output_path}")
