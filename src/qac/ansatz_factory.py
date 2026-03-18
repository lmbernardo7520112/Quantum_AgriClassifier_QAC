"""QAC — Ansatz Factory.

Creates variational ansatz circuits from string configuration.
Factory pattern mirrors AnsatzFactory in vqe_knapsack_local.
"""
from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, RealAmplitudes


class AnsatzFactory:
    """Factory for variational ansatz circuits."""

    VALID_TYPES = ("real_amplitudes", "efficient_su2")

    @staticmethod
    def create(
        ansatz_type: str,
        n_qubits: int,
        reps: int = 2,
    ) -> QuantumCircuit:
        """Create a variational ansatz circuit.

        Args:
            ansatz_type: One of 'real_amplitudes', 'efficient_su2'.
            n_qubits: Number of qubits.
            reps: Number of ansatz repetitions.

        Returns:
            Parametrized ansatz circuit.

        Raises:
            ValueError: If unknown ansatz type.
        """
        if ansatz_type == "real_amplitudes":
            return RealAmplitudes(n_qubits, reps=reps)
        elif ansatz_type == "efficient_su2":
            return EfficientSU2(n_qubits, reps=reps)
        else:
            raise ValueError(
                f"Unknown ansatz type: '{ansatz_type}'. Valid: {AnsatzFactory.VALID_TYPES}"
            )
