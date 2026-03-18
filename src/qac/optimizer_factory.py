"""QAC — Optimizer Factory.

Creates Qiskit optimizers from string configuration.
Factory pattern mirrors OptimizerFactory in vqe_knapsack_local.
"""
from __future__ import annotations

from qiskit_algorithms.optimizers import COBYLA, SPSA, Optimizer


class OptimizerFactory:
    """Factory for Qiskit optimizers."""

    VALID_TYPES = ("cobyla", "spsa")

    @staticmethod
    def create(optimizer_type: str, max_iter: int = 100) -> Optimizer:
        """Create a Qiskit optimizer instance.

        Args:
            optimizer_type: One of 'cobyla', 'spsa'.
            max_iter: Maximum number of iterations.

        Returns:
            Configured optimizer instance.

        Raises:
            ValueError: If unknown optimizer type.
        """
        if optimizer_type == "cobyla":
            return COBYLA(maxiter=max_iter)
        elif optimizer_type == "spsa":
            return SPSA(maxiter=max_iter)
        else:
            raise ValueError(
                f"Unknown optimizer type: '{optimizer_type}'. "
                f"Valid: {OptimizerFactory.VALID_TYPES}"
            )
