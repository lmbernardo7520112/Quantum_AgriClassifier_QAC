"""
QAC — IBM Quantum Backend.

Integration with IBM Quantum Runtime for real hardware execution.
Token managed via environment variable IBM_QUANTUM_TOKEN.
"""

from __future__ import annotations

import os
import time
from typing import Any


def check_ibm_token() -> bool:
    """Check if IBM Quantum token is configured."""
    return bool(os.environ.get("IBM_QUANTUM_TOKEN"))


def get_available_backends() -> list[str]:
    """List available IBM Quantum backends."""
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        return []

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        backends = service.backends()
        return [b.name for b in backends]
    except Exception as e:
        return [f"Error: {e}"]


def deploy_to_ibm(
    circuit: Any,
    backend_name: str = "ibm_brisbane",
    shots: int = 1024,
) -> dict[str, Any]:
    """
    Deploy and execute a quantum circuit on IBM Quantum hardware.

    Args:
        circuit: Qiskit QuantumCircuit
        backend_name: IBM backend name
        shots: Number of measurement shots

    Returns:
        Dict with job_id, results, and execution info
    """
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        return {
            "error": True,
            "error_type": "IBM_TOKEN_NOT_CONFIGURED",
            "message": "Set IBM_QUANTUM_TOKEN environment variable",
        }

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        backend = service.backend(backend_name)

        # Transpile and run
        from qiskit import transpile

        transpiled = transpile(circuit, backend=backend)

        sampler = SamplerV2(backend=backend)
        job = sampler.run([transpiled], shots=shots)

        start_time = time.time()
        result = job.result()
        execution_time = time.time() - start_time

        return {
            "job_id": job.job_id(),
            "ibm_backend_used": backend_name,
            "shots": shots,
            "execution_time_s": round(execution_time, 3),
            "result_counts": dict(result[0].data.meas.get_counts()) if result else {},
            "status": "COMPLETED",
        }
    except Exception as e:
        return {
            "error": True,
            "error_type": "IBM_EXECUTION_ERROR",
            "message": str(e),
            "ibm_backend_used": backend_name,
        }
