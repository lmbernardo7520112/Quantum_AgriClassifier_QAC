"""
QAC — Quantum SVM (QSVM).

Implements Quantum SVM using FidelityQuantumKernel with Qiskit V2 Primitives.
Supports Aer simulator and IBM Quantum backends.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.svm import SVC

from classical.baseline import compute_metrics
from classical.data_loader import DatasetResult
from quantum.feature_map import create_feature_map, normalize_features


def train_qsvm(
    dataset: DatasetResult,
    n_qubits: int = 8,
    feature_map_type: str = "zz",
    seed: int = 42,
    backend: str = "aer_statevector",
    models_dir: str | Path = "models",
    experiment_id: str = "",
) -> dict[str, Any]:
    """
    Train Quantum SVM using FidelityQuantumKernel.

    Uses V2 primitives (qiskit-machine-learning 0.8+).
    """
    from qiskit_aer import AerSimulator
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Ensure features match qubit count
    if dataset.feature_dim != n_qubits:
        raise ValueError(
            f"Feature dimension ({dataset.feature_dim}) must match n_qubits ({n_qubits}). "
            f"Apply PCA first with n_components={n_qubits}."
        )

    # Normalize features for quantum encoding
    X_train = normalize_features(dataset.X_train)
    X_test = normalize_features(dataset.X_test)

    # Create feature map
    feature_map = create_feature_map(n_qubits, feature_map_type)

    # Create quantum kernel
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    # Compute kernel matrices
    start_time = time.time()

    # Train SVM with precomputed quantum kernel
    kernel_train = quantum_kernel.evaluate(X_train)
    svm = SVC(kernel="precomputed", random_state=seed, probability=True)
    svm.fit(kernel_train, dataset.y_train)

    training_time = time.time() - start_time

    # Evaluate
    infer_start = time.time()
    kernel_test = quantum_kernel.evaluate(X_test, X_train)
    y_pred = svm.predict(kernel_test)
    inference_time = time.time() - infer_start

    metrics = compute_metrics(dataset.y_test, y_pred)
    metrics["training_time_s"] = round(training_time, 3)
    metrics["inference_time_s"] = round(inference_time, 3)

    # Save model
    model_data = {
        "svm": svm,
        "quantum_kernel_params": {
            "feature_map_type": feature_map_type,
            "n_qubits": n_qubits,
        },
        "X_train_normalized": X_train,
    }
    model_filename = f"qsvm_{experiment_id}.pkl"
    model_path = models_dir / model_filename
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    model_hash = _file_hash(str(model_path))

    return {
        "model_type": "qsvm",
        "model_path": str(model_path),
        "model_hash": model_hash,
        "metrics": metrics,
        "circuit_depth": feature_map.depth() if callable(feature_map.depth) else feature_map.depth,
        "n_qubits_used": n_qubits,
        "hyperparameters": {
            "feature_map": feature_map_type,
            "n_qubits": n_qubits,
            "backend": backend,
            "seed": seed,
        },
        "dataset_info": dataset.metadata,
    }


def _file_hash(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
