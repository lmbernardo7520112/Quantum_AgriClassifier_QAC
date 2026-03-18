"""QAC — Feature Map Factory.

Creates quantum feature map circuits from string configuration.
Factory pattern mirrors AnsatzFactory in vqe_knapsack_local.
"""
from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap


class FeatureMapFactory:
    """Factory for quantum feature map circuits."""

    VALID_TYPES = ("zz", "z", "pauli")

    @staticmethod
    def create(
        feature_map_type: str,
        n_qubits: int,
        reps: int = 2,
    ) -> QuantumCircuit:
        """Create a quantum feature map circuit.

        Args:
            feature_map_type: One of 'zz', 'z', 'pauli'.
            n_qubits: Number of qubits (must be ≤ 10).
            reps: Number of repetitions.

        Returns:
            Qiskit FeatureMap circuit.

        Raises:
            ValueError: If n_qubits > 10 or unknown type.
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
            raise ValueError(
                f"Unknown feature map type: '{feature_map_type}'. "
                f"Valid: {FeatureMapFactory.VALID_TYPES}"
            )
