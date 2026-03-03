"""
QAC — Tool Implementations.

Registers all tool functions with the MCP server.
Each tool receives: input_data, experiment_id, context, resource_registry, context_manager.
Tools are fully isolated (Invariant 4).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp_server.context_manager import ContextManager
from mcp_server.resource_registry import ResourceRegistry


# Dataset path defaults
DEFAULT_DATASET_ROOT = os.environ.get(
    "QAC_DATASET_ROOT",
    r"C:\Users\USER\Downloads\Quantum_AgriClassifier_QAC_dataset",
)
DEFAULT_PROJECT_ROOT = os.environ.get(
    "QAC_PROJECT_ROOT",
    str(Path(__file__).parent),
)


def tool_initialize_project(
    input_data: dict[str, Any],
    experiment_id: str,
    context: dict[str, Any],
    resource_registry: ResourceRegistry,
    context_manager: ContextManager,
) -> dict[str, Any]:
    """tool.initialize_project — Scaffold and validate environment."""
    project_root = Path(input_data.get("project_root", DEFAULT_PROJECT_ROOT))
    dataset_root = Path(input_data.get("dataset_root", DEFAULT_DATASET_ROOT))

    # Verify directories
    dirs_to_check = ["mcp_server", "classical", "quantum", "registry", "tasks", "tests", "models", "experiments"]
    for d in dirs_to_check:
        (project_root / d).mkdir(parents=True, exist_ok=True)

    # Check datasets
    datasets_found = []
    if (dataset_root / "EuroSAT_RGB.zip").exists() or (dataset_root / "EuroSAT_RGB").is_dir():
        datasets_found.append("eurosat_rgb")
    if (dataset_root / "EuroSAT_MS.zip").exists() or (dataset_root / "EuroSAT_MS").is_dir():
        datasets_found.append("eurosat_ms")
    if (dataset_root / "PlantVillage-Dataset").is_dir():
        datasets_found.append("plantvillage")

    return {
        "experiment_id": experiment_id,
        "status": "success",
        "datasets_found": datasets_found,
        "environment": {
            "project_root": str(project_root),
            "dataset_root": str(dataset_root),
        },
    }


def tool_load_dataset(
    input_data: dict[str, Any],
    experiment_id: str,
    context: dict[str, Any],
    resource_registry: ResourceRegistry,
    context_manager: ContextManager,
) -> dict[str, Any]:
    """tool.load_dataset — Load, hash, split, register dataset."""
    from classical.data_loader import load_dataset

    dataset_name = input_data["dataset_name"]
    dataset_path = input_data.get("dataset_path")
    seed = input_data.get("seed", 42)
    max_samples = input_data.get("max_samples")

    # Determine paths
    if not dataset_path:
        root = Path(DEFAULT_DATASET_ROOT)
        if dataset_name == "plantvillage":
            dataset_path = str(root / "PlantVillage-Dataset")
        elif dataset_name == "eurosat_rgb":
            dataset_path = str(root / "EuroSAT_RGB")
        elif dataset_name == "eurosat_ms":
            dataset_path = str(root / "EuroSAT_MS")

    result = load_dataset(dataset_name, dataset_path, seed=seed, max_samples=max_samples)

    # Register as resource.dataset
    resource = resource_registry.register(
        resource_type="resource.dataset",
        metadata={
            "dataset_name": dataset_name,
            "dataset_hash": result.dataset_hash,
            "num_classes": result.num_classes,
            "class_names": result.class_names,
            "splits": result.to_dict()["splits"],
            **result.metadata,
        },
        experiment_id=experiment_id,
    )

    # Store dataset hash in context
    context_manager.update_context(experiment_id, {"dataset_hash": result.dataset_hash})

    # Cache dataset for pipeline use
    _dataset_cache[resource["resource_id"]] = result

    return {
        "experiment_id": experiment_id,
        "resource": {
            "resource_type": "resource.dataset",
            "resource_id": resource["resource_id"],
        },
        "dataset_hash": result.dataset_hash,
        "num_classes": result.num_classes,
        "class_names": result.class_names,
        "splits": result.to_dict()["splits"],
    }


def tool_run_baseline(
    input_data: dict[str, Any],
    experiment_id: str,
    context: dict[str, Any],
    resource_registry: ResourceRegistry,
    context_manager: ContextManager,
) -> dict[str, Any]:
    """tool.run_baseline — Train classical baseline."""
    from classical.baseline import train_svm, train_cnn

    dataset_resource_id = input_data["dataset_resource_id"]
    model_type = input_data["model_type"]
    seed = input_data.get("seed", 42)

    dataset = _get_cached_dataset(dataset_resource_id)
    models_dir = Path(DEFAULT_PROJECT_ROOT) / "models"

    if model_type == "svm":
        result = train_svm(dataset, seed=seed, models_dir=models_dir, experiment_id=experiment_id)
    elif model_type == "cnn":
        result = train_cnn(dataset, seed=seed, models_dir=models_dir, experiment_id=experiment_id)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Register model
    model_resource = resource_registry.register(
        resource_type="resource.model",
        metadata={"model_type": model_type, "hyperparameters": result.get("hyperparameters", {})},
        file_path=result["model_path"],
        file_hash=result["model_hash"],
        experiment_id=experiment_id,
    )

    # Register metrics
    metrics_resource = resource_registry.register(
        resource_type="resource.metrics",
        metadata={"model_type": model_type, "metrics": result["metrics"]},
        experiment_id=experiment_id,
    )

    return {
        "experiment_id": experiment_id,
        "model_resource": {"resource_type": "resource.model", "resource_id": model_resource["resource_id"]},
        "metrics_resource": {"resource_type": "resource.metrics", "resource_id": metrics_resource["resource_id"]},
        "metrics": result["metrics"],
    }


def tool_train_qsvm(
    input_data: dict[str, Any],
    experiment_id: str,
    context: dict[str, Any],
    resource_registry: ResourceRegistry,
    context_manager: ContextManager,
) -> dict[str, Any]:
    """tool.train_qsvm — Quantum SVM training."""
    from classical.data_loader import apply_pca
    from quantum.qsvm import train_qsvm

    dataset_resource_id = input_data["dataset_resource_id"]
    n_qubits = input_data.get("n_qubits", 8)
    seed = input_data.get("seed", 42)

    dataset = _get_cached_dataset(dataset_resource_id)

    # PCA if needed
    if dataset.feature_dim != n_qubits:
        dataset = apply_pca(dataset, n_components=n_qubits, seed=seed)

    models_dir = Path(DEFAULT_PROJECT_ROOT) / "models"
    result = train_qsvm(dataset, n_qubits=n_qubits, seed=seed, models_dir=models_dir, experiment_id=experiment_id)

    model_resource = resource_registry.register(
        resource_type="resource.model",
        metadata={"model_type": "qsvm", "hyperparameters": result.get("hyperparameters", {})},
        file_path=result["model_path"],
        file_hash=result["model_hash"],
        experiment_id=experiment_id,
    )
    metrics_resource = resource_registry.register(
        resource_type="resource.metrics",
        metadata={"model_type": "qsvm", "metrics": result["metrics"]},
        experiment_id=experiment_id,
    )

    return {
        "experiment_id": experiment_id,
        "model_resource": {"resource_type": "resource.model", "resource_id": model_resource["resource_id"]},
        "metrics_resource": {"resource_type": "resource.metrics", "resource_id": metrics_resource["resource_id"]},
        "metrics": result["metrics"],
        "circuit_depth": result.get("circuit_depth", 0),
        "n_qubits_used": n_qubits,
    }


def tool_train_vqc(
    input_data: dict[str, Any],
    experiment_id: str,
    context: dict[str, Any],
    resource_registry: ResourceRegistry,
    context_manager: ContextManager,
) -> dict[str, Any]:
    """tool.train_vqc — VQC training."""
    from classical.data_loader import apply_pca
    from quantum.vqc import train_vqc

    dataset_resource_id = input_data["dataset_resource_id"]
    n_qubits = input_data.get("n_qubits", 8)
    seed = input_data.get("seed", 42)

    dataset = _get_cached_dataset(dataset_resource_id)
    if dataset.feature_dim != n_qubits:
        dataset = apply_pca(dataset, n_components=n_qubits, seed=seed)

    models_dir = Path(DEFAULT_PROJECT_ROOT) / "models"
    result = train_vqc(
        dataset, n_qubits=n_qubits, seed=seed,
        ansatz_type=input_data.get("ansatz", "real_amplitudes"),
        optimizer_type=input_data.get("optimizer", "cobyla"),
        max_iter=input_data.get("max_iter", 100),
        models_dir=models_dir, experiment_id=experiment_id,
    )

    model_resource = resource_registry.register(
        resource_type="resource.model",
        metadata={"model_type": "vqc", "hyperparameters": result.get("hyperparameters", {})},
        file_path=result["model_path"],
        file_hash=result["model_hash"],
        experiment_id=experiment_id,
    )
    metrics_resource = resource_registry.register(
        resource_type="resource.metrics",
        metadata={"model_type": "vqc", "metrics": result["metrics"]},
        experiment_id=experiment_id,
    )

    return {
        "experiment_id": experiment_id,
        "model_resource": {"resource_type": "resource.model", "resource_id": model_resource["resource_id"]},
        "metrics_resource": {"resource_type": "resource.metrics", "resource_id": metrics_resource["resource_id"]},
        "metrics": result["metrics"],
        "training_history": result.get("training_history", []),
        "n_qubits_used": n_qubits,
    }


def tool_train_data_reupload(
    input_data: dict[str, Any],
    experiment_id: str,
    context: dict[str, Any],
    resource_registry: ResourceRegistry,
    context_manager: ContextManager,
) -> dict[str, Any]:
    """tool.train_data_reupload — Data Re-uploading training."""
    from classical.data_loader import apply_pca
    from quantum.data_reupload import train_data_reupload

    dataset_resource_id = input_data["dataset_resource_id"]
    n_qubits = input_data.get("n_qubits", 8)
    seed = input_data.get("seed", 42)

    dataset = _get_cached_dataset(dataset_resource_id)
    if dataset.feature_dim != n_qubits:
        dataset = apply_pca(dataset, n_components=n_qubits, seed=seed)

    models_dir = Path(DEFAULT_PROJECT_ROOT) / "models"
    result = train_data_reupload(
        dataset, n_qubits=n_qubits, seed=seed,
        n_layers=input_data.get("n_layers", 3),
        models_dir=models_dir, experiment_id=experiment_id,
    )

    model_resource = resource_registry.register(
        resource_type="resource.model",
        metadata={"model_type": "data_reupload", "hyperparameters": result.get("hyperparameters", {})},
        file_path=result["model_path"],
        file_hash=result["model_hash"],
        experiment_id=experiment_id,
    )
    metrics_resource = resource_registry.register(
        resource_type="resource.metrics",
        metadata={"model_type": "data_reupload", "metrics": result["metrics"]},
        experiment_id=experiment_id,
    )

    return {
        "experiment_id": experiment_id,
        "model_resource": {"resource_type": "resource.model", "resource_id": model_resource["resource_id"]},
        "metrics_resource": {"resource_type": "resource.metrics", "resource_id": metrics_resource["resource_id"]},
        "metrics": result["metrics"],
    }


def tool_evaluate_model(
    input_data: dict[str, Any],
    experiment_id: str,
    context: dict[str, Any],
    resource_registry: ResourceRegistry,
    context_manager: ContextManager,
) -> dict[str, Any]:
    """tool.evaluate_model — Evaluate registered model."""
    from classical.baseline import evaluate_model

    model_resource_id = input_data["model_resource_id"]
    dataset_resource_id = input_data["dataset_resource_id"]
    split = input_data.get("split", "test")

    # Get model resource
    model = resource_registry.get("resource.model", model_resource_id)
    if not model:
        raise ValueError(f"Model resource not found: {model_resource_id}")

    dataset = _get_cached_dataset(dataset_resource_id)
    model_type = model["metadata"].get("model_type", "svm")

    metrics = evaluate_model(model["file_path"], model_type, dataset, split)

    metrics_resource = resource_registry.register(
        resource_type="resource.metrics",
        metadata={"model_type": model_type, "metrics": metrics, "split": split},
        experiment_id=experiment_id,
    )

    return {
        "experiment_id": experiment_id,
        "metrics_resource": {"resource_type": "resource.metrics", "resource_id": metrics_resource["resource_id"]},
        "metrics": metrics,
    }


def tool_compare_models(
    input_data: dict[str, Any],
    experiment_id: str,
    context: dict[str, Any],
    resource_registry: ResourceRegistry,
    context_manager: ContextManager,
) -> dict[str, Any]:
    """tool.compare_models — Cross-model comparison."""
    from classical.baseline import compare_models

    model_resource_ids = input_data["model_resource_ids"]

    metrics_list = []
    for rid in model_resource_ids:
        model = resource_registry.get("resource.model", rid)
        if model:
            # Find associated metrics
            all_metrics = resource_registry.list_resources("resource.metrics")
            model_metrics = [m for m in all_metrics if m.get("experiment_id") == model.get("experiment_id")]
            if model_metrics:
                metrics_list.append({
                    "model_type": model["metadata"].get("model_type", "unknown"),
                    "metrics": model_metrics[0]["metadata"].get("metrics", {}),
                    "resource_id": rid,
                })

    comparison = compare_models(metrics_list)

    return {
        "experiment_id": experiment_id,
        "comparison_table": comparison["comparison_table"],
        "best_model": comparison.get("best_model"),
        "quantum_vs_baseline": comparison.get("quantum_vs_baseline", {}),
    }


def tool_simulate_noise(
    input_data: dict[str, Any],
    experiment_id: str,
    context: dict[str, Any],
    resource_registry: ResourceRegistry,
    context_manager: ContextManager,
) -> dict[str, Any]:
    """tool.simulate_noise — NISQ noise simulation."""
    from quantum.noise import simulate_noise

    model_resource_id = input_data["model_resource_id"]
    dataset_resource_id = input_data["dataset_resource_id"]

    model = resource_registry.get("resource.model", model_resource_id)
    if not model:
        raise ValueError(f"Model not found: {model_resource_id}")

    dataset = _get_cached_dataset(dataset_resource_id)

    result = simulate_noise(
        model_result=model["metadata"],
        dataset_result=dataset,
        noise_type=input_data.get("noise_type", "combined"),
        noise_params=input_data.get("noise_params"),
        seed=input_data.get("seed", 42),
    )

    metrics_resource = resource_registry.register(
        resource_type="resource.metrics",
        metadata={
            "noise_simulation": True,
            "metrics_noiseless": result["metrics_noiseless"],
            "metrics_noisy": result["metrics_noisy"],
            "degradation_pct": result["degradation_pct"],
        },
        experiment_id=experiment_id,
    )

    return {
        "experiment_id": experiment_id,
        "metrics_noiseless": result["metrics_noiseless"],
        "metrics_noisy": result["metrics_noisy"],
        "degradation_pct": result["degradation_pct"],
        "noise_config": result["noise_config"],
    }


def tool_deploy_ibm(
    input_data: dict[str, Any],
    experiment_id: str,
    context: dict[str, Any],
    resource_registry: ResourceRegistry,
    context_manager: ContextManager,
) -> dict[str, Any]:
    """tool.deploy_ibm — IBM Quantum hardware deployment."""
    from quantum.ibm_backend import check_ibm_token, deploy_to_ibm

    if not check_ibm_token():
        return {
            "experiment_id": experiment_id,
            "error": True,
            "error_type": "IBM_TOKEN_NOT_CONFIGURED",
            "message": "Set IBM_QUANTUM_TOKEN environment variable.",
        }

    # Build a simple test circuit from the model
    from quantum.feature_map import create_feature_map
    n_qubits = input_data.get("n_qubits", 4)
    circuit = create_feature_map(n_qubits, "zz")

    result = deploy_to_ibm(
        circuit=circuit,
        backend_name=input_data.get("ibm_backend", "ibm_brisbane"),
        shots=input_data.get("shots", 1024),
    )

    if not result.get("error"):
        metrics_resource = resource_registry.register(
            resource_type="resource.metrics",
            metadata={"ibm_result": result},
            experiment_id=experiment_id,
        )

    return {
        "experiment_id": experiment_id,
        **result,
    }


# ─────────────────── Dataset Cache ─────────────────────

_dataset_cache: dict[str, Any] = {}


def _get_cached_dataset(resource_id: str):
    """Get dataset from cache. Raises if not found."""
    if resource_id not in _dataset_cache:
        raise ValueError(
            f"Dataset '{resource_id}' not found in cache. "
            f"Call tool.load_dataset first."
        )
    return _dataset_cache[resource_id]


# ─────────────────── Registration ──────────────────────

TOOL_FUNCTIONS = {
    "tool.initialize_project": tool_initialize_project,
    "tool.load_dataset": tool_load_dataset,
    "tool.run_baseline": tool_run_baseline,
    "tool.train_qsvm": tool_train_qsvm,
    "tool.train_vqc": tool_train_vqc,
    "tool.train_data_reupload": tool_train_data_reupload,
    "tool.evaluate_model": tool_evaluate_model,
    "tool.compare_models": tool_compare_models,
    "tool.simulate_noise": tool_simulate_noise,
    "tool.deploy_ibm": tool_deploy_ibm,
}


def register_all_tools(tool_registry) -> None:
    """Register all tool functions with the tool registry."""
    for name, func in TOOL_FUNCTIONS.items():
        tool_registry.register(name, func)
