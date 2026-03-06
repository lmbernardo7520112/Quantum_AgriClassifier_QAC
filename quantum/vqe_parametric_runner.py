"""
QAC — VQE Parametric Runner (Phase II).

Wraps `vqe_classifier.train_vqe_classifier()` for systematic parametric
experiments. Supports grid search over optimizer, ansatz depth, and
Hamiltonian parameters with multi-seed reproducibility.

This file is a NEW incremental extension — no existing files are modified.

Governed by: docs/VQE_PHASE2_EXPERIMENTAL_PROTOCOL.md
"""

from __future__ import annotations

import hashlib
import itertools
import json
import pickle
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from classical.data_loader import DatasetResult
from quantum.feature_map import normalize_features
from quantum.hamiltonian_builder import build_ising_hamiltonian
from quantum.vqe_classifier import train_vqe_classifier


# ─────────────────── Data Classes ──────────────────────


@dataclass
class ParametricConfig:
    """Single parametric configuration for a VQE experiment."""

    optimizer_type: str = "cobyla"
    ansatz_type: str = "real_amplitudes"
    ansatz_reps: int = 2
    coupling_strength: float = 0.5
    transverse_field: float = 0.5
    max_iter: int = 100
    n_qubits: int = 8
    backend: str = "aer_statevector"

    @property
    def config_id(self) -> str:
        """Deterministic config identifier."""
        return (
            f"{self.optimizer_type}_reps{self.ansatz_reps}"
            f"_g{self.coupling_strength}_h{self.transverse_field}"
        )


@dataclass
class ParametricResult:
    """Result of a single parametric VQE experiment."""

    config: ParametricConfig
    seed: int
    experiment_id: str
    accuracy: float = 0.0
    f1_score: float = 0.0
    delta_E: float = 0.0
    var_intra_class0: float = 0.0
    var_intra_class1: float = 0.0
    cohens_d: float = 0.0
    p_value_ttest: float = 1.0
    p_value_permutation: float = 1.0
    training_time_s: float = 0.0
    inference_time_s: float = 0.0
    vqe_energies: dict[str, float] = field(default_factory=dict)
    model_path: str = ""
    model_hash: str = ""
    convergence: bool = False
    status: str = "PENDING"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON persistence."""
        d = asdict(self)
        d["config"] = asdict(self.config)
        return d


# ─────────────────── Core Runner ───────────────────────


def run_vqe_parametric_experiment(
    dataset: DatasetResult,
    config: ParametricConfig,
    seed: int,
    experiment_id: str,
    models_dir: str | Path = "models",
) -> ParametricResult:
    """
    Run a single VQE parametric experiment.

    Wraps `train_vqe_classifier()` from `quantum/vqe_classifier.py`
    and extends with additional statistical metrics.

    Args:
        dataset: DatasetResult with PCA-reduced features (feature_dim == n_qubits)
        config: Parametric configuration
        seed: Random seed for this run
        experiment_id: Unique experiment identifier
        models_dir: Directory to save model artifacts

    Returns:
        ParametricResult with full metrics
    """
    result = ParametricResult(
        config=config,
        seed=seed,
        experiment_id=experiment_id,
    )
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Map optimizer_type to what vqe_classifier accepts
        # L-BFGS-B uses "cobyla" path internally but we override via scipy
        optimizer_for_vqe = config.optimizer_type
        if config.optimizer_type == "L-BFGS-B":
            # vqe_classifier uses scipy.optimize.minimize;
            # "cobyla" path triggers COBYLA, "spsa" triggers Nelder-Mead.
            # For L-BFGS-B, we use the custom runner below.
            vqe_result = _run_vqe_with_lbfgsb(
                dataset=dataset,
                config=config,
                seed=seed,
                models_dir=models_dir,
                experiment_id=experiment_id,
            )
        else:
            # Standard path: reuse train_vqe_classifier directly
            vqe_result = train_vqe_classifier(
                dataset=dataset,
                n_qubits=config.n_qubits,
                ansatz_type=config.ansatz_type,
                optimizer_type=optimizer_for_vqe,
                max_iter=config.max_iter,
                coupling_strength=config.coupling_strength,
                transverse_field=config.transverse_field,
                seed=seed,
                backend=config.backend,
                models_dir=models_dir,
                experiment_id=experiment_id,
            )

        # Extract base metrics
        metrics = vqe_result["metrics"]
        result.accuracy = metrics["accuracy"]
        result.f1_score = metrics["f1_score"]
        result.training_time_s = metrics.get("training_time_s", 0.0)
        result.inference_time_s = metrics.get("inference_time_s", 0.0)
        result.vqe_energies = vqe_result.get("vqe_energies", {})
        result.model_path = vqe_result.get("model_path", "")
        result.model_hash = vqe_result.get("model_hash", "")

        # Check convergence
        result.convergence = all(
            float(v) < 0 for v in result.vqe_energies.values()
        )

        # Compute extended statistics
        _compute_extended_statistics(
            result=result,
            dataset=dataset,
            config=config,
            model_path=result.model_path,
            seed=seed,
        )

        result.status = "COMPLETED"

    except Exception as e:
        result.status = "FAILED"
        result.error = str(e)

    return result


def _run_vqe_with_lbfgsb(
    dataset: DatasetResult,
    config: ParametricConfig,
    seed: int,
    models_dir: Path,
    experiment_id: str,
) -> dict[str, Any]:
    """
    Run VQE with L-BFGS-B optimizer.

    Uses the same architecture as vqe_classifier.py but with L-BFGS-B.
    This avoids modifying the original file.
    """
    from qiskit.circuit.library import EfficientSU2, RealAmplitudes
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp
    from scipy.optimize import minimize as scipy_minimize

    from classical.baseline import compute_metrics
    from quantum.hamiltonian_builder import (
        build_all_class_hamiltonians,
        build_ising_hamiltonian,
        hamiltonian_metadata,
    )

    np.random.seed(seed)

    X_train = normalize_features(dataset.X_train)
    X_test = normalize_features(dataset.X_test)

    # Build per-class Hamiltonians
    class_hamiltonians = build_all_class_hamiltonians(
        X_train, dataset.y_train, config.n_qubits,
        coupling_strength=config.coupling_strength,
        transverse_field=config.transverse_field,
    )

    # Create ansatz with configurable reps
    if config.ansatz_type == "real_amplitudes":
        ansatz = RealAmplitudes(config.n_qubits, reps=config.ansatz_reps)
    elif config.ansatz_type == "efficient_su2":
        ansatz = EfficientSU2(config.n_qubits, reps=config.ansatz_reps)
    else:
        raise ValueError(f"Unknown ansatz: {config.ansatz_type}")

    n_params = ansatz.num_parameters
    estimator = StatevectorEstimator(seed=seed)

    # Train: VQE per class via L-BFGS-B
    start_time = time.time()
    class_results = {}
    training_history = []

    for cls_label, hamiltonian in class_hamiltonians.items():
        eval_count = [0]

        def cost_fn(params, _h=hamiltonian):
            eval_count[0] += 1
            pub = (ansatz, [_h], [params])
            job = estimator.run([pub])
            return float(job.result()[0].data.evs[0])

        def grad_fn(params, _h=hamiltonian, eps=1e-4):
            """Finite-difference gradient for L-BFGS-B."""
            grad = np.zeros_like(params)
            f0 = cost_fn(params, _h)
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += eps
                grad[i] = (cost_fn(params_plus, _h) - f0) / eps
            return grad

        x0 = np.random.uniform(-np.pi, np.pi, n_params)

        opt_result = scipy_minimize(
            cost_fn, x0, method="L-BFGS-B",
            jac=grad_fn,
            options={"maxiter": config.max_iter, "ftol": 1e-10},
        )

        optimal_energy = float(opt_result.fun)
        optimal_params = opt_result.x.tolist()

        class_results[cls_label] = {
            "optimal_params": None,
            "optimal_value": optimal_energy,
            "optimal_point": optimal_params,
        }
        training_history.append({
            "class": cls_label,
            "energy": optimal_energy,
            "evals": eval_count[0],
        })

    training_time = time.time() - start_time

    # Inference
    infer_start = time.time()
    y_pred = []

    for x in X_test:
        sample_hamiltonian = build_ising_hamiltonian(
            x, config.n_qubits,
            coupling_strength=config.coupling_strength,
            transverse_field=config.transverse_field,
        )
        energies = {}
        for cls_label, cls_data in class_results.items():
            bound_circuit = ansatz.assign_parameters(cls_data["optimal_point"])
            pub = (bound_circuit, [sample_hamiltonian])
            job = estimator.run([pub])
            energy = float(job.result()[0].data.evs[0])
            energies[cls_label] = energy
        pred_class = min(energies, key=energies.get)
        y_pred.append(pred_class)

    inference_time = time.time() - infer_start
    y_pred = np.array(y_pred)

    metrics = compute_metrics(dataset.y_test, y_pred)
    metrics["training_time_s"] = round(training_time, 3)
    metrics["inference_time_s"] = round(inference_time, 3)

    vqe_energies = {str(k): v["optimal_value"] for k, v in class_results.items()}
    metrics["vqe_energies"] = vqe_energies
    metrics["mean_vqe_energy"] = float(np.mean(list(vqe_energies.values())))

    # Save model
    model_filename = f"vqe_classifier_{experiment_id}.pkl"
    model_path = models_dir / model_filename
    model_data = {
        "class_results": {
            k: {"optimal_point": v["optimal_point"], "optimal_value": v["optimal_value"]}
            for k, v in class_results.items()
        },
        "hyperparameters": {
            "ansatz": config.ansatz_type,
            "optimizer": "L-BFGS-B",
            "max_iter": config.max_iter,
            "n_qubits": config.n_qubits,
            "ansatz_reps": config.ansatz_reps,
            "coupling_strength": config.coupling_strength,
            "transverse_field": config.transverse_field,
            "backend": config.backend,
            "seed": seed,
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
        "n_qubits_used": config.n_qubits,
        "n_classes": len(class_results),
        "training_history": training_history,
        "hyperparameters": model_data["hyperparameters"],
        "dataset_info": dataset.metadata,
    }


def _compute_extended_statistics(
    result: ParametricResult,
    dataset: DatasetResult,
    config: ParametricConfig,
    model_path: str,
    seed: int,
) -> None:
    """
    Compute extended statistics: ΔE, intra-class variance, Cohen's d,
    t-test p-value, permutation p-value.

    Updates `result` in place.
    """
    from qiskit.circuit.library import RealAmplitudes, EfficientSU2
    from qiskit.primitives import StatevectorEstimator

    if not model_path or not Path(model_path).exists():
        return

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    X_test_norm = normalize_features(dataset.X_test)

    if config.ansatz_type == "real_amplitudes":
        ansatz = RealAmplitudes(config.n_qubits, reps=config.ansatz_reps)
    else:
        ansatz = EfficientSU2(config.n_qubits, reps=config.ansatz_reps)

    estimator = StatevectorEstimator(seed=seed)

    # Compute per-sample energies for each class
    energies_by_class = {0: [], 1: []}

    for x in X_test_norm:
        H_x = build_ising_hamiltonian(
            x, config.n_qubits,
            coupling_strength=config.coupling_strength,
            transverse_field=config.transverse_field,
        )
        for cls in [0, 1]:
            theta = model_data["class_results"][cls]["optimal_point"]
            bound = ansatz.assign_parameters(theta)
            pub = (bound, [H_x])
            job = estimator.run([pub])
            e = float(job.result()[0].data.evs[0])
            energies_by_class[cls].append(e)

    E0 = np.array(energies_by_class[0])
    E1 = np.array(energies_by_class[1])

    # Split by true labels
    E_when_y0 = E0[dataset.y_test == 0]
    E_when_y1 = E1[dataset.y_test == 1]

    # ΔE
    result.delta_E = abs(float(np.mean(E0) - np.mean(E1)))

    # Intra-class variance
    result.var_intra_class0 = float(np.var(E_when_y0)) if len(E_when_y0) > 0 else 0.0
    result.var_intra_class1 = float(np.var(E_when_y1)) if len(E_when_y1) > 0 else 0.0

    # Cohen's d
    pooled_std = np.sqrt(
        (np.var(E_when_y0) + np.var(E_when_y1)) / 2
    )
    if pooled_std > 0:
        result.cohens_d = abs(float(np.mean(E_when_y0) - np.mean(E_when_y1))) / pooled_std
    else:
        result.cohens_d = 0.0

    # Welch's t-test
    if len(E_when_y0) > 1 and len(E_when_y1) > 1:
        _, p_ttest = stats.ttest_ind(E_when_y0, E_when_y1, equal_var=False)
        result.p_value_ttest = float(p_ttest)
    else:
        result.p_value_ttest = 1.0

    # Permutation test (1000 iterations)
    if len(E_when_y0) > 0 and len(E_when_y1) > 0:
        E_all = np.concatenate([E_when_y0, E_when_y1])
        T_obs = abs(np.mean(E_when_y0) - np.mean(E_when_y1))
        n0 = len(E_when_y0)

        rng = np.random.RandomState(seed)
        T_null = []
        for _ in range(1000):
            perm = rng.permutation(E_all)
            T_null.append(abs(np.mean(perm[:n0]) - np.mean(perm[n0:])))
        T_null = np.array(T_null)

        result.p_value_permutation = float((np.sum(T_null >= T_obs) + 1) / (1000 + 1))
    else:
        result.p_value_permutation = 1.0


# ─────────────────── Grid Runner ───────────────────────


def build_default_grid() -> list[ParametricConfig]:
    """
    Build the default parametric grid for Phase II.

    12 configurations covering:
    - 3 optimizers (COBYLA, SPSA, L-BFGS-B)
    - 3 ansatz depths (reps ∈ {2, 4, 6})
    - 3 coupling strengths (γ ∈ {0.1, 0.5, 1.0})
    - 3 transverse fields (h ∈ {0.1, 0.5, 1.0})
    """
    return [
        # Baseline depth sweep (COBYLA, varying reps)
        ParametricConfig(optimizer_type="cobyla", ansatz_reps=2, coupling_strength=0.5, transverse_field=0.5),
        ParametricConfig(optimizer_type="cobyla", ansatz_reps=4, coupling_strength=0.5, transverse_field=0.5),
        ParametricConfig(optimizer_type="cobyla", ansatz_reps=6, coupling_strength=0.5, transverse_field=0.5),
        # Hamiltonian parameter sweep (COBYLA, reps=4)
        ParametricConfig(optimizer_type="cobyla", ansatz_reps=4, coupling_strength=0.1, transverse_field=0.5),
        ParametricConfig(optimizer_type="cobyla", ansatz_reps=4, coupling_strength=1.0, transverse_field=0.5),
        ParametricConfig(optimizer_type="cobyla", ansatz_reps=4, coupling_strength=0.5, transverse_field=0.1),
        ParametricConfig(optimizer_type="cobyla", ansatz_reps=4, coupling_strength=0.5, transverse_field=1.0),
        # Optimizer comparison (SPSA)
        ParametricConfig(optimizer_type="spsa", ansatz_reps=2, coupling_strength=0.5, transverse_field=0.5),
        ParametricConfig(optimizer_type="spsa", ansatz_reps=4, coupling_strength=0.5, transverse_field=0.5),
        # Optimizer comparison (L-BFGS-B)
        ParametricConfig(optimizer_type="L-BFGS-B", ansatz_reps=4, coupling_strength=0.5, transverse_field=0.5),
        ParametricConfig(optimizer_type="L-BFGS-B", ansatz_reps=6, coupling_strength=0.5, transverse_field=0.5),
        # High expressivity + strong Hamiltonian
        ParametricConfig(optimizer_type="L-BFGS-B", ansatz_reps=6, coupling_strength=1.0, transverse_field=1.0),
    ]


def run_parametric_grid(
    dataset: DatasetResult,
    grid: list[ParametricConfig] | None = None,
    seeds: list[int] | None = None,
    models_dir: str | Path = "models",
    results_dir: str | Path = "experiments/vqe_phase2",
    phase_id: str = "VQE_PHASE2",
    log_fn: Any = None,
) -> list[ParametricResult]:
    """
    Run the full parametric grid with multiple seeds.

    Args:
        dataset: DatasetResult with PCA-reduced features
        grid: List of ParametricConfig (default: 12 configs)
        seeds: List of seeds per config (default: [42, 123, 7])
        models_dir: Directory for model artifacts
        results_dir: Directory for experiment result JSONs
        phase_id: Phase identifier prefix for experiment IDs
        log_fn: Optional logging function

    Returns:
        List of ParametricResult for all experiments
    """
    if grid is None:
        grid = build_default_grid()
    if seeds is None:
        seeds = [42, 123, 7]
    if log_fn is None:
        log_fn = lambda msg: print(f"[VQE-P2] {msg}")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[ParametricResult] = []
    total = len(grid) * len(seeds)
    run_idx = 0

    for config in grid:
        for seed in seeds:
            run_idx += 1
            experiment_id = f"{phase_id}_{config.config_id}_s{seed}"

            log_fn(f"[{run_idx}/{total}] {experiment_id}")
            log_fn(f"  optimizer={config.optimizer_type}, reps={config.ansatz_reps}, "
                    f"γ={config.coupling_strength}, h={config.transverse_field}, seed={seed}")

            result = run_vqe_parametric_experiment(
                dataset=dataset,
                config=config,
                seed=seed,
                experiment_id=experiment_id,
                models_dir=models_dir,
            )

            all_results.append(result)

            # Persist individual result
            result_file = results_dir / f"{experiment_id}.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

            if result.status == "COMPLETED":
                log_fn(f"  ✅ acc={result.accuracy:.4f}, f1={result.f1_score:.4f}, "
                        f"ΔE={result.delta_E:.4f}, p={result.p_value_ttest:.4f}")
            else:
                log_fn(f"  ❌ FAILED: {result.error}")

    # Save consolidated results
    consolidated_file = results_dir / "consolidated_results.json"
    with open(consolidated_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": phase_id,
                "total_experiments": len(all_results),
                "completed": sum(1 for r in all_results if r.status == "COMPLETED"),
                "failed": sum(1 for r in all_results if r.status == "FAILED"),
                "grid_size": len(grid),
                "seeds": seeds,
                "results": [r.to_dict() for r in all_results],
            },
            f, indent=2, ensure_ascii=False,
        )

    log_fn(f"Consolidated results saved: {consolidated_file}")
    return all_results


# ─────────────────── Utilities ─────────────────────────


def _file_hash(filepath: str) -> str:
    """SHA-256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
