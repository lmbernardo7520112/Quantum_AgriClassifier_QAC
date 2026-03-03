"""
QAC — Variational Quantum Classifier (VQC).

Implements VQC from qiskit-machine-learning with configurable ansatz
and optimizer. Supports COBYLA and SPSA optimizers.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from classical.baseline import compute_metrics
from classical.data_loader import DatasetResult
from quantum.feature_map import create_feature_map, normalize_features


def train_vqc(
    dataset: DatasetResult,
    n_qubits: int = 8,
    ansatz_type: str = "real_amplitudes",
    optimizer_type: str = "cobyla",
    max_iter: int = 100,
    seed: int = 42,
    backend: str = "aer_statevector",
    models_dir: str | Path = "models",
    experiment_id: str = "",
) -> dict[str, Any]:
    """
    Train Variational Quantum Classifier.

    Args:
        dataset: DatasetResult with PCA-reduced features
        n_qubits: Number of qubits (must match feature_dim)
        ansatz_type: 'real_amplitudes' or 'efficient_su2'
        optimizer_type: 'cobyla' or 'spsa'
        max_iter: Maximum optimizer iterations
        seed: Random seed
    """
    from qiskit.circuit.library import EfficientSU2, RealAmplitudes
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit_machine_learning.algorithms import VQC

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if dataset.feature_dim != n_qubits:
        raise ValueError(
            f"Feature dimension ({dataset.feature_dim}) must match n_qubits ({n_qubits}). "
            f"Apply PCA first."
        )

    # Normalize
    X_train = normalize_features(dataset.X_train)
    X_test = normalize_features(dataset.X_test)

    # Feature map
    feature_map = create_feature_map(n_qubits, "zz")

    # Ansatz
    if ansatz_type == "real_amplitudes":
        ansatz = RealAmplitudes(n_qubits, reps=2)
    elif ansatz_type == "efficient_su2":
        ansatz = EfficientSU2(n_qubits, reps=2)
    else:
        raise ValueError(f"Unknown ansatz: {ansatz_type}")

    # Optimizer
    if optimizer_type == "cobyla":
        optimizer = COBYLA(maxiter=max_iter)
    elif optimizer_type == "spsa":
        optimizer = SPSA(maxiter=max_iter)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # Training history callback
    history = []

    def callback(weights, obj_func_eval):
        history.append({"iteration": len(history), "objective": float(obj_func_eval)})

    # Create and train VQC
    start_time = time.time()

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
    )

    # One-hot encode labels for VQC
    num_classes = dataset.num_classes
    y_train_onehot = np.eye(num_classes)[dataset.y_train]

    vqc.fit(X_train, y_train_onehot)
    training_time = time.time() - start_time

    # Evaluate
    infer_start = time.time()
    y_pred_onehot = vqc.predict(X_test)
    y_pred = np.argmax(y_pred_onehot, axis=1) if y_pred_onehot.ndim > 1 else y_pred_onehot
    inference_time = time.time() - infer_start

    metrics = compute_metrics(dataset.y_test, y_pred)
    metrics["training_time_s"] = round(training_time, 3)
    metrics["inference_time_s"] = round(inference_time, 3)

    # Save
    model_filename = f"vqc_{experiment_id}.pkl"
    model_path = models_dir / model_filename
    model_data = {
        "weights": vqc.weights if hasattr(vqc, "weights") else None,
        "hyperparameters": {
            "ansatz": ansatz_type,
            "optimizer": optimizer_type,
            "max_iter": max_iter,
            "n_qubits": n_qubits,
        },
        "training_history": history,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    model_hash = _file_hash(str(model_path))

    return {
        "model_type": "vqc",
        "model_path": str(model_path),
        "model_hash": model_hash,
        "metrics": metrics,
        "training_history": history,
        "n_qubits_used": n_qubits,
        "hyperparameters": {
            "ansatz": ansatz_type,
            "optimizer": optimizer_type,
            "max_iter": max_iter,
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
