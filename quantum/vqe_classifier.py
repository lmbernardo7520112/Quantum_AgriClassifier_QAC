"""
QAC — VQE Classifier.

Implements VQE-based quantum classification using data-conditioned
Ising Hamiltonians. Each class gets its own optimal ansatz parameters
via energy minimization.

Training: For each class c, VQE finds θ*_c minimizing ⟨ψ(θ)|H_c|ψ(θ)⟩
Inference: Classify x as argmin_c ⟨ψ(θ*_c)|H(x)|ψ(θ*_c)⟩

Uses qiskit.primitives.StatevectorEstimator (exact, no transpilation needed).
"""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize as scipy_minimize
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from classical.baseline import compute_metrics
from classical.data_loader import DatasetResult
from quantum.feature_map import normalize_features
from quantum.hamiltonian_builder import (
    build_all_class_hamiltonians,
    build_ising_hamiltonian,
    hamiltonian_metadata,
)


def _evaluate_energy(
    params: np.ndarray,
    ansatz,
    hamiltonian: SparsePauliOp,
    estimator: StatevectorEstimator,
) -> float:
    """Evaluate ⟨ψ(θ)|H|ψ(θ)⟩ for given parameters."""
    pub = (ansatz, [hamiltonian], [params])
    job = estimator.run([pub])
    result = job.result()
    return float(result[0].data.evs[0])


def _evaluate_energy_bound(
    bound_circuit,
    hamiltonian: SparsePauliOp,
    estimator: StatevectorEstimator,
) -> float:
    """Evaluate ⟨ψ|H|ψ⟩ for a bound (parameter-free) circuit."""
    pub = (bound_circuit, [hamiltonian])
    job = estimator.run([pub])
    result = job.result()
    return float(result[0].data.evs[0])


def train_vqe_classifier(
    dataset: DatasetResult,
    n_qubits: int = 8,
    ansatz_type: str = "real_amplitudes",
    optimizer_type: str = "cobyla",
    max_iter: int = 100,
    coupling_strength: float = 1.0,
    transverse_field: float = 0.5,
    seed: int = 42,
    backend: str = "aer_statevector",
    models_dir: str | Path = "models",
    experiment_id: str = "",
) -> dict[str, Any]:
    """
    Train VQE-based classifier.

    Workflow:
    1. Build per-class Hamiltonians from training centroids
    2. Run VQE per class to find optimal θ*_c (via scipy.optimize)
    3. Classify test samples by minimum energy evaluation

    Args:
        dataset: DatasetResult with PCA-reduced features
        n_qubits: Number of qubits (must match feature_dim)
        ansatz_type: 'real_amplitudes' or 'efficient_su2'
        optimizer_type: 'cobyla' or 'spsa'
        max_iter: Maximum optimizer iterations
        coupling_strength: Ising ZZ coupling factor
        transverse_field: Ising X field factor
        seed: Random seed for reproducibility
        backend: Quantum backend identifier (logged, not instantiated)
        models_dir: Directory to save model artifacts
        experiment_id: Experiment identifier

    Returns:
        Dict with model_path, model_hash, metrics, VQE energies
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if dataset.feature_dim != n_qubits:
        raise ValueError(
            f"Feature dimension ({dataset.feature_dim}) must match n_qubits ({n_qubits}). "
            f"Apply PCA first."
        )

    np.random.seed(seed)

    # Normalize features
    X_train = normalize_features(dataset.X_train)
    X_test = normalize_features(dataset.X_test)

    # Build per-class Hamiltonians
    class_hamiltonians = build_all_class_hamiltonians(
        X_train, dataset.y_train, n_qubits,
        coupling_strength=coupling_strength,
        transverse_field=transverse_field,
    )

    # Create ansatz
    if ansatz_type == "real_amplitudes":
        ansatz = RealAmplitudes(n_qubits, reps=2)
    elif ansatz_type == "efficient_su2":
        ansatz = EfficientSU2(n_qubits, reps=2)
    else:
        raise ValueError(f"Unknown ansatz: {ansatz_type}")

    n_params = ansatz.num_parameters

    # Create estimator (exact statevector — no Aer needed)
    estimator = StatevectorEstimator(seed=seed)

    # Train: VQE per class via scipy.optimize
    start_time = time.time()
    class_results = {}
    training_history = []

    print(f"  [VQE] Training {len(class_hamiltonians)} classes, {n_params} params each...")

    for cls_label, hamiltonian in class_hamiltonians.items():
        print(f"  [VQE] Optimizing class {cls_label}...")

        # Cost function for scipy
        eval_count = [0]

        def cost_fn(params, _h=hamiltonian):
            eval_count[0] += 1
            return _evaluate_energy(params, ansatz, _h, estimator)

        # Random initial parameters
        x0 = np.random.uniform(-np.pi, np.pi, n_params)

        # Run scipy optimizer
        if optimizer_type == "cobyla":
            opt_result = scipy_minimize(
                cost_fn, x0, method="COBYLA",
                options={"maxiter": max_iter, "rhobeg": 0.5},
            )
        elif optimizer_type == "spsa":
            # Fallback to Nelder-Mead for gradient-free
            opt_result = scipy_minimize(
                cost_fn, x0, method="Nelder-Mead",
                options={"maxiter": max_iter},
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        optimal_energy = float(opt_result.fun)
        optimal_params = opt_result.x.tolist()

        class_results[cls_label] = {
            "optimal_params": None,  # raw ParameterView not serializable
            "optimal_value": optimal_energy,
            "optimal_point": optimal_params,
        }
        training_history.append({
            "class": cls_label,
            "energy": optimal_energy,
            "evals": eval_count[0],
        })
        print(f"  [VQE] Class {cls_label}: E*={optimal_energy:.6f} ({eval_count[0]} evals)")

    training_time = time.time() - start_time

    # Inference: evaluate each test sample against all class Hamiltonians
    infer_start = time.time()
    y_pred = []
    print(f"  [VQE] Inference on {len(X_test)} test samples...")

    for x_idx, x in enumerate(X_test):
        sample_hamiltonian = build_ising_hamiltonian(
            x, n_qubits,
            coupling_strength=coupling_strength,
            transverse_field=transverse_field,
        )

        energies = {}
        for cls_label, cls_data in class_results.items():
            # Bind optimal params to ansatz and evaluate energy
            bound_circuit = ansatz.assign_parameters(cls_data["optimal_point"])
            energy = _evaluate_energy_bound(bound_circuit, sample_hamiltonian, estimator)
            energies[cls_label] = energy

        # Classify as class with minimum energy
        pred_class = min(energies, key=energies.get)
        y_pred.append(pred_class)

        if (x_idx + 1) % 25 == 0:
            print(f"  [VQE] {x_idx + 1}/{len(X_test)} inferred")

    inference_time = time.time() - infer_start
    y_pred = np.array(y_pred)

    # Compute metrics
    metrics = compute_metrics(dataset.y_test, y_pred)
    metrics["training_time_s"] = round(training_time, 3)
    metrics["inference_time_s"] = round(inference_time, 3)

    # Extra VQE-specific metrics
    vqe_energies = {
        str(k): v["optimal_value"] for k, v in class_results.items()
    }
    metrics["vqe_energies"] = vqe_energies
    metrics["mean_vqe_energy"] = float(np.mean(list(vqe_energies.values())))

    # Save model artifact
    model_filename = f"vqe_classifier_{experiment_id}.pkl"
    model_path = models_dir / model_filename
    model_data = {
        "class_results": {
            k: {"optimal_point": v["optimal_point"], "optimal_value": v["optimal_value"]}
            for k, v in class_results.items()
        },
        "hyperparameters": {
            "ansatz": ansatz_type,
            "optimizer": optimizer_type,
            "max_iter": max_iter,
            "n_qubits": n_qubits,
            "coupling_strength": coupling_strength,
            "transverse_field": transverse_field,
            "backend": backend,
            "seed": seed,
        },
        "hamiltonian_metadata": {
            str(k): hamiltonian_metadata(v, k)
            for k, v in class_hamiltonians.items()
        },
        "training_history": training_history,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    model_hash = _file_hash(str(model_path))

    return {
        "model_type": "vqe_classifier",
        "model_path": str(model_path),
        "model_hash": model_hash,
        "metrics": metrics,
        "vqe_energies": vqe_energies,
        "n_qubits_used": n_qubits,
        "n_classes": len(class_results),
        "training_history": training_history,
        "hyperparameters": model_data["hyperparameters"],
        "dataset_info": dataset.metadata,
    }


def _file_hash(filepath: str) -> str:
    """SHA-256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
