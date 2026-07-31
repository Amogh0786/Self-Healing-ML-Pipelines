"""
Unit tests for mathematical drift detection (KL Divergence & PSI).
"""
import numpy as np
import pytest
from src.drift.detector import calculate_kl_divergence, calculate_psi


def test_kl_divergence_identical():
    """Identical probability distributions must have KL divergence of 0.0."""
    p = np.array([0.2, 0.3, 0.5])
    kl = calculate_kl_divergence(p, p.copy())
    assert abs(kl) < 1e-9, f"Expected 0.0 KL divergence for identical distributions, got {kl}"


def test_kl_divergence_shifted():
    """Shifted distributions must have strictly positive KL divergence."""
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.2, 0.7])
    kl = calculate_kl_divergence(p, q)
    assert kl > 0.5, f"Expected high KL divergence for inverted distribution, got {kl}"


def test_psi_identical():
    """Identical distributions must have PSI of 0.0."""
    p = np.array([0.25, 0.25, 0.5])
    psi = calculate_psi(p, p.copy())
    assert abs(psi) < 1e-9, f"Expected 0.0 PSI for identical distributions, got {psi}"


def test_psi_symmetric_behavior():
    """PSI is symmetric: PSI(P, Q) == PSI(Q, P)."""
    p = np.array([0.8, 0.1, 0.1])
    q = np.array([0.2, 0.3, 0.5])
    psi_pq = calculate_psi(p, q)
    psi_qp = calculate_psi(q, p)
    assert abs(psi_pq - psi_qp) < 1e-9, "PSI must be symmetric between P and Q"
