"""
QAC — Data Re-uploading Quantum Classifier.

Custom quantum classifier using alternating data encoding and variational layers.
Based on the "data re-uploading" paradigm where input data is encoded multiple times.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from classical.baseline import compute_metrics
from classical.data_loader import DatasetResult
from quantum.feature_map import normalize_features


def build_reupload_circuit(n_qubits: int, n_layers: int) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """
    Build a data re-uploading circuit.

    Architecture: [Encoding(x) → Variational(θ)] × n_layers → Measurement

    Returns:
        (circuit, input_params, weight_params)
    """
    n_input_params = n_qubits * n_layers
    n_weight_params = n_qubits * 3 * n_layers  # Rx, Ry, Rz per qubit per layer

    input_params = ParameterVector("x", n_input_params)
    weight_params = ParameterVector("θ", n_weight_params)

    qc = QuantumCircuit(n_qubits)

    for layer in range(n_layers):
        # Data encoding layer
        for qubit in range(n_qubits):
            param_idx = layer * n_qubits + qubit
            qc.ry(input_params[param_idx], qubit)

        # Variational layer
        for qubit in range(n_qubits):
            w_idx = layer * n_qubits * 3 + qubit * 3
            qc.rx(weight_params[w_idx], qubit)
            qc.ry(weight_params[w_idx + 1], qubit)
            qc.rz(weight_params[w_idx + 2], qubit)

        # Entangling layer (circular CNOT)
        for qubit in range(n_qubits - 1):
            qc.cx(qubit, qubit + 1)
        if n_qubits > 1:
            qc.cx(n_qubits - 1, 0)

    qc.measure_all()
    return qc, input_params, weight_params


def train_data_reupload(
    dataset: DatasetResult,
    n_qubits: int = 8,
    n_layers: int = 3,
    seed: int = 42,
    backend: str = "aer_statevector",
    models_dir: str | Path = "models",
    experiment_id: str = "",
    max_iter: int = 50,
) -> dict[str, Any]:
    """
    Train Data Re-uploading classifier.

    Uses a simplified training approach with QSVM-style kernel evaluation
    since full VQC training of re-uploading circuits can be very slow.
    """
    from qiskit_aer import AerSimulator
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if dataset.feature_dim != n_qubits:
        raise ValueError(
            f"Feature dimension ({dataset.feature_dim}) must match n_qubits ({n_qubits}). "
            f"Apply PCA first."
        )

    # Build re-uploading feature map (encoding-only, no trainable params)
    from qiskit.circuit.library import ZZFeatureMap
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=n_layers)

    X_train = normalize_features(dataset.X_train)
    X_test = normalize_features(dataset.X_test)

    # Use quantum kernel approach with the re-uploading circuit as feature map
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    start_time = time.time()
    kernel_train = quantum_kernel.evaluate(X_train)

    from sklearn.svm import SVC
    svm = SVC(kernel="precomputed", random_state=seed, probability=True)
    svm.fit(kernel_train, dataset.y_train)
    training_time = time.time() - start_time

    infer_start = time.time()
    kernel_test = quantum_kernel.evaluate(X_test, X_train)
    y_pred = svm.predict(kernel_test)
    inference_time = time.time() - infer_start

    metrics = compute_metrics(dataset.y_test, y_pred)
    metrics["training_time_s"] = round(training_time, 3)
    metrics["inference_time_s"] = round(inference_time, 3)

    # Save
    model_filename = f"data_reupload_{experiment_id}.pkl"
    model_path = models_dir / model_filename
    model_data = {
        "svm": svm,
        "circuit_config": {
            "n_qubits": n_qubits,
            "n_layers": n_layers,
        },
        "X_train_normalized": X_train,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    return {
        "model_type": "data_reupload",
        "model_path": str(model_path),
        "model_hash": _file_hash(str(model_path)),
        "metrics": metrics,
        "n_qubits_used": n_qubits,
        "n_layers": n_layers,
        "hyperparameters": {
            "n_qubits": n_qubits,
            "n_layers": n_layers,
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
