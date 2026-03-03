"""
QAC — NISQ Noise Simulation.

Applies noise models from Qiskit Aer to quantum classifiers
and measures degradation vs noiseless baseline.

Supports: depolarizing, thermal relaxation, readout error, combined.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)

from classical.baseline import compute_metrics


def create_noise_model(
    noise_type: str = "combined",
    params: dict[str, Any] | None = None,
) -> NoiseModel:
    """
    Create a Qiskit Aer NoiseModel.

    Args:
        noise_type: 'depolarizing', 'thermal', 'readout', or 'combined'
        params: Noise parameters (defaults provided)
    """
    params = params or {}
    noise_model = NoiseModel()

    if noise_type in ("depolarizing", "combined"):
        p1 = params.get("depolarizing_1q", 0.001)
        p2 = params.get("depolarizing_2q", 0.01)
        error_1q = depolarizing_error(p1)
        error_2q = depolarizing_error(p2, 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ["rx", "ry", "rz", "h"])
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    if noise_type in ("thermal", "combined"):
        t1 = params.get("t1", 50e3)  # ns
        t2 = params.get("t2", 70e3)  # ns
        gate_time_1q = params.get("gate_time_1q", 50)  # ns
        gate_time_2q = params.get("gate_time_2q", 300)  # ns

        error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
        error_2q = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(
            thermal_relaxation_error(t1, t2, gate_time_2q)
        )
        noise_model.add_all_qubit_quantum_error(error_1q, ["rx", "ry", "rz"])
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    if noise_type in ("readout", "combined"):
        p0_1 = params.get("readout_p0_1", 0.015)  # P(1|0)
        p1_0 = params.get("readout_p1_0", 0.015)  # P(0|1)
        readout = ReadoutError([[1 - p0_1, p0_1], [p1_0, 1 - p1_0]])
        noise_model.add_all_qubit_readout_error(readout)

    return noise_model


def simulate_noise(
    model_result: dict[str, Any],
    dataset_result: Any,
    noise_type: str = "combined",
    noise_params: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run noisy simulation and measure degradation.

    Args:
        model_result: Result dict from quantum training (contains model_type, hyperparameters)
        dataset_result: DatasetResult
        noise_type: Type of noise to apply
        noise_params: Noise parameters
        seed: Random seed

    Returns:
        Dict with noiseless metrics, noisy metrics, and degradation percentage
    """
    from quantum.feature_map import normalize_features, create_feature_map
    from qiskit_aer import AerSimulator
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    import pickle
    from sklearn.svm import SVC

    noise_model = create_noise_model(noise_type, noise_params)

    # Get model params
    n_qubits = model_result.get("n_qubits_used", 8)
    feature_map_type = model_result.get("hyperparameters", {}).get("feature_map", "zz")
    n_layers = model_result.get("hyperparameters", {}).get("n_layers", 2)

    feature_map = create_feature_map(n_qubits, feature_map_type, reps=n_layers)

    X_train = normalize_features(dataset_result.X_train)
    X_test = normalize_features(dataset_result.X_test)

    # Noiseless evaluation (baseline)
    kernel_noiseless = FidelityQuantumKernel(feature_map=feature_map)
    K_train_clean = kernel_noiseless.evaluate(X_train)
    svm_clean = SVC(kernel="precomputed", random_state=seed)
    svm_clean.fit(K_train_clean, dataset_result.y_train)
    K_test_clean = kernel_noiseless.evaluate(X_test, X_train)
    y_pred_clean = svm_clean.predict(K_test_clean)
    metrics_noiseless = compute_metrics(dataset_result.y_test, y_pred_clean)

    # Noisy evaluation
    noisy_backend = AerSimulator(noise_model=noise_model)
    kernel_noisy = FidelityQuantumKernel(
        feature_map=feature_map,
    )
    K_train_noisy = kernel_noisy.evaluate(X_train)
    svm_noisy = SVC(kernel="precomputed", random_state=seed)
    svm_noisy.fit(K_train_noisy, dataset_result.y_train)
    K_test_noisy = kernel_noisy.evaluate(X_test, X_train)
    y_pred_noisy = svm_noisy.predict(K_test_noisy)
    metrics_noisy = compute_metrics(dataset_result.y_test, y_pred_noisy)

    # Compute degradation
    acc_clean = metrics_noiseless["accuracy"]
    acc_noisy = metrics_noisy["accuracy"]
    degradation = ((acc_clean - acc_noisy) / acc_clean * 100) if acc_clean > 0 else 0

    return {
        "metrics_noiseless": metrics_noiseless,
        "metrics_noisy": metrics_noisy,
        "degradation_pct": round(degradation, 2),
        "noise_config": {
            "noise_type": noise_type,
            "noise_params": noise_params or {},
        },
        "nisq_limitation": degradation > 30,
    }
