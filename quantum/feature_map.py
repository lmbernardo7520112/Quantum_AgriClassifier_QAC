"""
QAC — Quantum Feature Maps.

Provides quantum feature map circuits for encoding classical data
into quantum states. Configurable for ≤10 qubits with PCA preprocessing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap


def create_feature_map(
    n_qubits: int,
    feature_map_type: str = "zz",
    reps: int = 2,
) -> Any:
    """
    Create a quantum feature map circuit.

    Args:
        n_qubits: Number of qubits (must be ≤ 10)
        feature_map_type: 'zz', 'z', or 'pauli'
        reps: Number of repetitions

    Returns:
        Qiskit FeatureMap circuit
    """
    if n_qubits > 10:
        raise ValueError(f"n_qubits must be ≤ 10, got {n_qubits}")

    if feature_map_type == "zz":
        return ZZFeatureMap(feature_dimension=n_qubits, reps=reps)
    elif feature_map_type == "z":
        return ZFeatureMap(feature_dimension=n_qubits, reps=reps)
    elif feature_map_type == "pauli":
        return PauliFeatureMap(feature_dimension=n_qubits, reps=reps)
    else:
        raise ValueError(f"Unknown feature map type: {feature_map_type}. Use 'zz', 'z', or 'pauli'.")


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Normalize features to [0, 2π] range for quantum encoding."""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1  # avoid division by zero
    return (X - X_min) / X_range * 2 * np.pi
