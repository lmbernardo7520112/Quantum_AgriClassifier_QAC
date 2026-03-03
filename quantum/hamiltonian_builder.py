"""
QAC — Hamiltonian Builder.

Constructs data-conditioned Ising Hamiltonians for VQE-based
quantum classification. Each data sample x is encoded into a
parameterized Hamiltonian H(x).

Supports:
- Ising model with transverse field
- Data-dependent coupling strengths
- SparsePauliOp output (Qiskit 1.x native)

NOTE: This is a STRUCTURAL addition — no execution, no backend instantiation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def build_ising_hamiltonian(
    features: np.ndarray,
    n_qubits: int,
    coupling_strength: float = 1.0,
    transverse_field: float = 0.5,
) -> SparsePauliOp:
    """
    Build a data-conditioned Ising Hamiltonian.

    H(x) = Σᵢ xᵢ Zᵢ + coupling * Σ_{i<j} xᵢxⱼ ZᵢZⱼ + field * Σᵢ Xᵢ

    Args:
        features: 1D array of PCA-reduced features (length = n_qubits)
        n_qubits: Number of qubits
        coupling_strength: Scaling factor for ZZ interactions
        transverse_field: Scaling factor for X terms

    Returns:
        SparsePauliOp representing H(x)
    """
    if len(features) != n_qubits:
        raise ValueError(
            f"Feature length ({len(features)}) must match n_qubits ({n_qubits})"
        )

    pauli_list = []
    coeffs = []

    # Single-qubit Z terms: xᵢ Zᵢ
    for i in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - 1 - i] = "Z"  # Qiskit uses reverse qubit ordering
        pauli_list.append("".join(label))
        coeffs.append(float(features[i]))

    # Two-qubit ZZ terms: coupling * xᵢxⱼ ZᵢZⱼ
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            label = ["I"] * n_qubits
            label[n_qubits - 1 - i] = "Z"
            label[n_qubits - 1 - j] = "Z"
            pauli_list.append("".join(label))
            coeffs.append(coupling_strength * float(features[i] * features[j]))

    # Transverse field X terms: field * Xᵢ
    for i in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - 1 - i] = "X"
        pauli_list.append("".join(label))
        coeffs.append(transverse_field)

    return SparsePauliOp.from_list(list(zip(pauli_list, coeffs))).simplify()


def build_class_hamiltonian(
    X_class: np.ndarray,
    n_qubits: int,
    coupling_strength: float = 1.0,
    transverse_field: float = 0.5,
) -> SparsePauliOp:
    """
    Build Hamiltonian from class centroid.

    Computes the centroid of all samples belonging to a class,
    then constructs the Ising Hamiltonian from the centroid vector.

    Args:
        X_class: Array of shape (n_samples, n_qubits) — all samples for one class
        n_qubits: Number of qubits
        coupling_strength: ZZ coupling factor
        transverse_field: X field factor

    Returns:
        SparsePauliOp representing H(centroid)
    """
    centroid = X_class.mean(axis=0)
    return build_ising_hamiltonian(
        centroid, n_qubits,
        coupling_strength=coupling_strength,
        transverse_field=transverse_field,
    )


def build_all_class_hamiltonians(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    coupling_strength: float = 1.0,
    transverse_field: float = 0.5,
) -> dict[int, SparsePauliOp]:
    """
    Build class-conditioned Hamiltonians for all classes.

    Args:
        X: Feature matrix (n_samples, n_qubits)
        y: Label array (n_samples,)
        n_qubits: Number of qubits

    Returns:
        Dict mapping class_label → SparsePauliOp
    """
    unique_classes = np.unique(y)
    hamiltonians = {}

    for cls in unique_classes:
        mask = y == cls
        X_class = X[mask]
        hamiltonians[int(cls)] = build_class_hamiltonian(
            X_class, n_qubits,
            coupling_strength=coupling_strength,
            transverse_field=transverse_field,
        )

    return hamiltonians


def hamiltonian_metadata(
    hamiltonian: SparsePauliOp,
    class_label: int,
) -> dict[str, Any]:
    """Extract metadata from a Hamiltonian for registry storage."""
    return {
        "class_label": class_label,
        "num_qubits": hamiltonian.num_qubits,
        "num_terms": len(hamiltonian),
        "pauli_labels": [str(p) for p in hamiltonian.paulis][:10],  # first 10 for brevity
    }
