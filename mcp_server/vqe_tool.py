"""
QAC — VQE Tool Schema and Implementation (Additive Extension).

This module ADDS the tool.train_vqe_classifier to the MCP server
WITHOUT modifying any existing schemas or tool implementations.

To activate, import and call register_vqe_tool() after register_all_tools().
"""

from __future__ import annotations

from typing import Any

# ─────────────── Schema Definition ─────────────────────

VQE_TOOL_SCHEMA = {
    "tool.train_vqe_classifier": {
        "name": "tool.train_vqe_classifier",
        "description": "Train VQE-based quantum classifier using data-conditioned Ising Hamiltonians.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_resource_id": {"type": "string"},
                "n_qubits": {
                    "type": "integer",
                    "minimum": 2,
                    "maximum": 10,
                },
                "ansatz": {
                    "type": "string",
                    "enum": ["real_amplitudes", "efficient_su2"],
                },
                "optimizer": {
                    "type": "string",
                    "enum": ["cobyla", "spsa"],
                },
                "max_iter": {"type": "integer", "minimum": 1},
                "coupling_strength": {"type": "number", "minimum": 0},
                "transverse_field": {"type": "number", "minimum": 0},
                "seed": {"type": "integer"},
                "backend": {
                    "type": "string",
                    "enum": ["aer_statevector", "aer_qasm"],
                },
                "n_components_pca": {"type": ["integer", "null"]},
            },
            "required": ["dataset_resource_id"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "experiment_id": {"type": "string"},
                "model_resource": {"type": "object"},
                "metrics_resource": {"type": "object"},
                "metrics": {"type": "object"},
                "vqe_energies": {"type": "object"},
                "n_qubits_used": {"type": "integer"},
                "n_classes": {"type": "integer"},
                "training_history": {"type": "array"},
            },
            "required": ["experiment_id", "model_resource", "metrics_resource", "metrics"],
        },
        "preconditions": ["dataset_loaded"],
        "postconditions": [
            "resource.model_registered",
            "resource.metrics_registered",
            "experiment_registered",
        ],
    },
}


# ─────────────── Tool Implementation ───────────────────

def tool_train_vqe_classifier(
    input_data: dict[str, Any],
    experiment_id: str,
    context: dict[str, Any],
    resource_registry: Any,
    context_manager: Any,
) -> dict[str, Any]:
    """tool.train_vqe_classifier — VQE-based quantum classification."""
    import os
    from pathlib import Path

    from classical.data_loader import apply_pca
    from quantum.vqe_classifier import train_vqe_classifier

    # Import the dataset cache from the main implementations module
    from mcp_server.tool_implementations import _get_cached_dataset

    dataset_resource_id = input_data["dataset_resource_id"]
    n_qubits = input_data.get("n_qubits", 8)
    seed = input_data.get("seed", 42)

    dataset = _get_cached_dataset(dataset_resource_id)
    if dataset.feature_dim != n_qubits:
        dataset = apply_pca(dataset, n_components=n_qubits, seed=seed)

    models_dir = Path(os.environ.get(
        "QAC_PROJECT_ROOT",
        str(Path(__file__).parent),
    )) / "models"

    result = train_vqe_classifier(
        dataset,
        n_qubits=n_qubits,
        seed=seed,
        ansatz_type=input_data.get("ansatz", "real_amplitudes"),
        optimizer_type=input_data.get("optimizer", "cobyla"),
        max_iter=input_data.get("max_iter", 100),
        coupling_strength=input_data.get("coupling_strength", 1.0),
        transverse_field=input_data.get("transverse_field", 0.5),
        models_dir=models_dir,
        experiment_id=experiment_id,
    )

    # Register model resource
    model_resource = resource_registry.register(
        resource_type="resource.model",
        metadata={
            "model_type": "vqe_classifier",
            "hyperparameters": result.get("hyperparameters", {}),
        },
        file_path=result["model_path"],
        file_hash=result["model_hash"],
        experiment_id=experiment_id,
    )

    # Register metrics resource
    metrics_resource = resource_registry.register(
        resource_type="resource.metrics",
        metadata={
            "model_type": "vqe_classifier",
            "metrics": result["metrics"],
            "vqe_energies": result.get("vqe_energies", {}),
        },
        experiment_id=experiment_id,
    )

    return {
        "experiment_id": experiment_id,
        "model_resource": {
            "resource_type": "resource.model",
            "resource_id": model_resource["resource_id"],
        },
        "metrics_resource": {
            "resource_type": "resource.metrics",
            "resource_id": metrics_resource["resource_id"],
        },
        "metrics": result["metrics"],
        "vqe_energies": result.get("vqe_energies", {}),
        "n_qubits_used": n_qubits,
        "n_classes": result.get("n_classes", 0),
        "training_history": result.get("training_history", []),
    }


# ─────────────── Registration ──────────────────────────

def register_vqe_tool(tool_registry, schema_registry=None) -> None:
    """
    Register VQE tool schema and implementation.

    Call this AFTER register_all_tools() to extend the MCP server.
    Does NOT modify any existing tools or schemas.

    Usage in server.py:
        from mcp_server.tool_implementations import register_all_tools
        register_all_tools(tool_registry)

        from mcp_server.vqe_tool import register_vqe_tool
        register_vqe_tool(tool_registry, schema_registry)
    """
    # Register schema if schema_registry provided
    if schema_registry is not None:
        schema_registry.register_tool(VQE_TOOL_SCHEMA["tool.train_vqe_classifier"])

    # Register tool function
    tool_registry.register("tool.train_vqe_classifier", tool_train_vqe_classifier)
