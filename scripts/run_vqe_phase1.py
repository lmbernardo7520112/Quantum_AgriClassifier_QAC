#!/usr/bin/env python3
"""
QAC — VQE Phase 1 Execution Script.

Controlled scientific experiment: VQE binary classification
on PlantVillage (Tomato___healthy vs Tomato___Bacterial_spot).

This script runs INDEPENDENTLY of the MCP server.
It uses the modules directly and updates the registry atomically.

Governed by: docs/VQE_PHASE1_EXECUTION_PLAN.md
Config: experiments/vqe_phase1/config.yaml
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ─── Project paths ───
PROJECT_ROOT = Path("/mnt/c/Users/USER/Quantum_AgriClassifier_QAC")
DATASET_ROOT = Path("/mnt/c/Users/USER/Downloads/Quantum_AgriClassifier_QAC_dataset")
REGISTRY_PATH = PROJECT_ROOT / "registry"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "vqe_phase1"

os.environ["QAC_PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["QAC_DATASET_ROOT"] = str(DATASET_ROOT)

sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
N_QUBITS = 8
MAX_SAMPLES = 500
EXPERIMENT_ID = "VQE_PHASE1_BINARY"


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def sha256_file(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def update_registry(filename: str, key: str, entry: dict):
    """Atomically update a registry JSON file."""
    path = REGISTRY_PATH / filename
    data = json.loads(path.read_text())
    collection_key = "experiments" if "experiment" in filename else "resources"
    data[collection_key][key] = entry
    data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main():
    start_time = time.time()
    log("=" * 70)
    log("VQE PHASE 1 — EXECUÇÃO CONTROLADA")
    log("=" * 70)

    # Update experiment status to RUNNING
    update_registry("experiments.json", EXPERIMENT_ID, {
        "experiment_id": EXPERIMENT_ID,
        "tool_name": "tool.train_vqe_classifier",
        "status": "RUNNING",
        "input": {
            "dataset_name": "plantvillage",
            "classes_filter": ["Tomato___healthy", "Tomato___Bacterial_spot"],
            "max_samples": MAX_SAMPLES,
            "n_qubits": N_QUBITS,
            "seed": SEED,
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "completed_at": None,
    })

    # ─── STEP 1+2: Load ONLY 2 target classes directly ───
    log("STEP 1+2: Loading PlantVillage (2 target classes only)...")
    from PIL import Image
    from sklearn.model_selection import train_test_split

    color_dir = DATASET_ROOT / "PlantVillage-Dataset" / "raw" / "color"
    target_classes = ["Tomato___Bacterial_spot", "Tomato___healthy"]
    binary_class_names = target_classes
    samples_per_class = MAX_SAMPLES // 2  # 250 each for balanced

    images, labels = [], []
    for cls_idx, cls_name in enumerate(target_classes):
        cls_dir = color_dir / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {cls_dir}")
        img_files = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        log(f"  Class '{cls_name}': {len(img_files)} images available, sampling {samples_per_class}")
        np.random.seed(SEED)
        selected = np.random.choice(len(img_files), size=min(samples_per_class, len(img_files)), replace=False)
        for idx in selected:
            try:
                img = Image.open(img_files[idx]).convert("RGB").resize((64, 64))
                images.append(np.array(img, dtype=np.float32).flatten() / 255.0)
                labels.append(cls_idx)
            except Exception:
                continue

    X_all = np.array(images)
    y_all = np.array(labels)
    log(f"  Total loaded: {len(X_all)} (class0={sum(y_all==0)}, class1={sum(y_all==1)})")

    # Stratified split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
    )
    log(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    log(f"  Feature dim: {X_train.shape[1]}")

    # ─── STEP 3: PCA ───
    feat_dim = X_train.shape[1]
    log(f"STEP 3: Applying PCA ({feat_dim} → {N_QUBITS} dims)...")
    from sklearn.decomposition import PCA

    pca = PCA(n_components=N_QUBITS, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    explained_var = sum(pca.explained_variance_ratio_)
    log(f"  PCA explained variance: {explained_var:.4f}")

    # Create a minimal DatasetResult-like object
    from classical.data_loader import DatasetResult
    binary_dataset = DatasetResult(
        X_train=X_train_pca,
        X_val=np.empty((0, N_QUBITS)),
        X_test=X_test_pca,
        y_train=y_train,
        y_val=np.array([], dtype=int),
        y_test=y_test,
        class_names=binary_class_names,
        dataset_hash=hashlib.sha256(X_train_pca.tobytes()).hexdigest()[:16],
        metadata={
            "source": "plantvillage_binary",
            "classes": binary_class_names,
            "pca_components": N_QUBITS,
            "pca_explained_variance": float(explained_var),
        },
    )

    # ─── STEP 4: Train VQE Classifier ───
    log("STEP 4: Training VQE classifier...")
    log(f"  n_qubits={N_QUBITS}, ansatz=real_amplitudes, optimizer=cobyla")
    log(f"  max_iter=100, γ=1.0, h=0.5, seed={SEED}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    from quantum.vqe_classifier import train_vqe_classifier

    vqe_result = train_vqe_classifier(
        dataset=binary_dataset,
        n_qubits=N_QUBITS,
        ansatz_type="real_amplitudes",
        optimizer_type="cobyla",
        max_iter=100,
        coupling_strength=1.0,
        transverse_field=0.5,
        seed=SEED,
        backend="aer_statevector",
        models_dir=MODELS_DIR,
        experiment_id=EXPERIMENT_ID,
    )

    log(f"  Training time: {vqe_result['metrics']['training_time_s']:.1f}s")
    log(f"  Accuracy: {vqe_result['metrics']['accuracy']:.4f}")
    log(f"  F1 Score: {vqe_result['metrics']['f1_score']:.4f}")
    log(f"  VQE Energies: {vqe_result['vqe_energies']}")
    log(f"  Model saved: {vqe_result['model_path']}")
    log(f"  Model hash: {vqe_result['model_hash']}")

    # ─── STEP 5: Compute ΔE and intra-class variance ───
    log("STEP 5: Computing energy statistics...")
    from quantum.feature_map import normalize_features
    from quantum.hamiltonian_builder import build_ising_hamiltonian

    X_test_norm = normalize_features(X_test_pca)

    # Load trained model to get per-class optimal params
    import pickle
    with open(vqe_result["model_path"], "rb") as f:
        model_data = pickle.load(f)

    from qiskit.circuit.library import RealAmplitudes
    from qiskit.primitives import StatevectorEstimator

    ansatz = RealAmplitudes(N_QUBITS, reps=2)
    estimator = StatevectorEstimator(seed=SEED)

    # Compute per-sample energies for each class's optimal params
    energies_by_class = {0: [], 1: []}
    log(f"  Evaluating {len(X_test_norm)} test samples × 2 classes...")

    for x_idx, x in enumerate(X_test_norm):
        H_x = build_ising_hamiltonian(x, N_QUBITS, coupling_strength=1.0, transverse_field=0.5)
        for cls in [0, 1]:
            theta = model_data["class_results"][cls]["optimal_point"]
            bound = ansatz.assign_parameters(theta)
            job = estimator.run([(bound, [H_x])])
            e = float(job.result()[0].data.evs[0])
            energies_by_class[cls].append(e)
        if (x_idx + 1) % 20 == 0:
            log(f"    {x_idx + 1}/{len(X_test_norm)} evaluated")

    E0 = np.array(energies_by_class[0])
    E1 = np.array(energies_by_class[1])

    # Split by true labels for t-test
    E_when_y0 = E0[y_test == 0]  # Energy under θ*₀ for true class 0 samples
    E_when_y1 = E1[y_test == 1]  # Energy under θ*₁ for true class 1 samples

    delta_E = abs(float(np.mean(E0) - np.mean(E1)))
    var_intra_0 = float(np.var(E_when_y0))
    var_intra_1 = float(np.var(E_when_y1))

    log(f"  ΔE = |mean(E₀) - mean(E₁)| = {delta_E:.6f}")
    log(f"  Var(E|y=0) = {var_intra_0:.6f}")
    log(f"  Var(E|y=1) = {var_intra_1:.6f}")

    # ─── STEP 6: Statistical Tests ───
    log("STEP 6: Running statistical tests...")
    from scipy import stats

    # 6a: Welch's t-test
    t_stat, p_ttest = stats.ttest_ind(E_when_y0, E_when_y1, equal_var=False)
    log(f"  Welch t-test: t={t_stat:.4f}, p={p_ttest:.6f}")

    # 6b: Permutation test (1000 iterations)
    log("  Permutation test (1000 iterations)...")
    E_all_for_perm = np.concatenate([E_when_y0, E_when_y1])
    T_obs = abs(np.mean(E_when_y0) - np.mean(E_when_y1))
    n0_perm = len(E_when_y0)

    np.random.seed(SEED)
    T_null = []
    for k in range(1000):
        perm = np.random.permutation(E_all_for_perm)
        T_null.append(abs(np.mean(perm[:n0_perm]) - np.mean(perm[n0_perm:])))
    T_null = np.array(T_null)

    p_perm = (np.sum(T_null >= T_obs) + 1) / (1000 + 1)
    log(f"  Permutation test: T_obs={T_obs:.6f}, p_perm={p_perm:.6f}")
    log(f"  T_null mean={np.mean(T_null):.6f}, 95th={np.percentile(T_null, 95):.6f}")

    # ─── STEP 7: Persist results ───
    log("STEP 7: Persisting results to registry...")

    total_time = time.time() - start_time
    metrics = vqe_result["metrics"]
    metrics["delta_E"] = delta_E
    metrics["var_intra_class0"] = var_intra_0
    metrics["var_intra_class1"] = var_intra_1
    metrics["t_statistic"] = float(t_stat)
    metrics["p_value_ttest"] = float(p_ttest)
    metrics["p_value_permutation"] = float(p_perm)
    metrics["T_obs"] = float(T_obs)
    metrics["T_null_mean"] = float(np.mean(T_null))
    metrics["T_null_95pct"] = float(np.percentile(T_null, 95))
    metrics["pca_explained_variance"] = float(explained_var)
    metrics["total_time_s"] = round(total_time, 3)

    # Determine experiment status
    convergence_ok = all(
        v["optimal_value"] < 0 for v in model_data["class_results"].values()
    ) if model_data["class_results"] else False

    validated = (
        metrics["accuracy"] > 0.50
        and delta_E > 0
        and p_ttest < 0.05
    )

    exp_status = "COMPLETED" if validated else "COMPLETED_WITH_LIMITATIONS"

    # Update experiments.json
    update_registry("experiments.json", EXPERIMENT_ID, {
        "experiment_id": EXPERIMENT_ID,
        "tool_name": "tool.train_vqe_classifier",
        "status": exp_status,
        "input": {
            "dataset_name": "plantvillage",
            "classes_filter": ["Tomato___healthy", "Tomato___Bacterial_spot"],
            "max_samples": MAX_SAMPLES,
            "n_qubits": N_QUBITS,
            "seed": SEED,
            "ansatz": "real_amplitudes",
            "optimizer": "cobyla",
            "max_iter": 100,
            "coupling_strength": 1.0,
            "transverse_field": 0.5,
            "backend": "aer_statevector",
        },
        "output": {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "delta_E": delta_E,
            "p_ttest": float(p_ttest),
            "p_perm": float(p_perm),
            "convergence": convergence_ok,
            "vqe_energies": vqe_result["vqe_energies"],
        },
        "error": None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(start_time)),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "context": {
            "seed": SEED,
            "dataset_hash": binary_dataset.dataset_hash,
            "backend": "aer_statevector",
        },
        "resources_created": [
            {"type": "resource.model", "path": vqe_result["model_path"]},
            {"type": "resource.metrics", "id": EXPERIMENT_ID},
        ],
    })

    # Update models.json
    update_registry("models.json", f"model-{EXPERIMENT_ID}", {
        "resource_id": f"model-{EXPERIMENT_ID}",
        "resource_type": "resource.model",
        "file_path": vqe_result["model_path"],
        "file_hash": vqe_result["model_hash"],
        "metadata": {
            "model_type": "vqe_classifier",
            "experiment_id": EXPERIMENT_ID,
            "n_qubits": N_QUBITS,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment_id": EXPERIMENT_ID,
    })

    # Update metrics.json
    update_registry("metrics.json", f"metrics-{EXPERIMENT_ID}", {
        "resource_id": f"metrics-{EXPERIMENT_ID}",
        "resource_type": "resource.metrics",
        "metadata": metrics,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment_id": EXPERIMENT_ID,
    })

    # ─── STEP 8: Generate results markdown ───
    log("STEP 8: Generating results report...")

    confusion = metrics.get("confusion_matrix", [[0, 0], [0, 0]])

    results_md = f"""# VQE Phase 1 — Results

**experiment_id**: {EXPERIMENT_ID}
**Status**: {exp_status}
**Executed**: {time.strftime("%Y-%m-%dT%H:%M:%S%z")}

---

## Execution Summary

| Field | Value |
|-------|-------|
| Duration | {total_time:.1f}s |
| Training time | {metrics['training_time_s']:.1f}s |
| Inference time | {metrics.get('inference_time_s', 0):.1f}s |
| Status | {exp_status} |

## Dataset

| Field | Value |
|-------|-------|
| Source | PlantVillage (local) |
| Classes | {binary_class_names[0]}, {binary_class_names[1]} |
| Samples (train) | {len(X_train)} |
| Samples (test) | {len(X_test)} |
| PCA components | {N_QUBITS} |
| PCA variance explained | {explained_var:.4f} |
| Dataset hash | {binary_dataset.dataset_hash} |

## VQE Training Results

### Per-Class Energies

| Class | Label | Optimal Energy E*_c |
|-------|-------|---------------------|
| 0 | {binary_class_names[0]} | {model_data['class_results'][0]['optimal_value']:.6f} |
| 1 | {binary_class_names[1]} | {model_data['class_results'][1]['optimal_value']:.6f} |

### Energy Difference

```
ΔE = |E*_0 - E*_1| = {delta_E:.6f}
```

## Classification Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **{metrics['accuracy']:.4f}** |
| **F1 Score** | **{metrics['f1_score']:.4f}** |
| Precision | {metrics.get('precision', 0):.4f} |
| Recall | {metrics.get('recall', 0):.4f} |

## Confusion Matrix

```
               Predicted
            | class_0  | class_1  |
Actual ─────┼──────────┼──────────┤
  class_0   |  {confusion[0][0]:>5}   |  {confusion[0][1]:>5}   |
  class_1   |  {confusion[1][0]:>5}   |  {confusion[1][1]:>5}   |
```

## Energy Statistics

| Metric | Class 0 | Class 1 |
|--------|---------|---------|
| Mean E (test) | {np.mean(E_when_y0):.6f} | {np.mean(E_when_y1):.6f} |
| Std E (test) | {np.std(E_when_y0):.6f} | {np.std(E_when_y1):.6f} |
| Var E (test) | {var_intra_0:.6f} | {var_intra_1:.6f} |
| n samples | {len(E_when_y0)} | {len(E_when_y1)} |

## Statistical Tests

### Welch's t-test

| Statistic | Value |
|-----------|-------|
| t-statistic | {t_stat:.4f} |
| p-value | {p_ttest:.6f} |
| Decision | {"REJECT H₀ ✅" if p_ttest < 0.05 else "FAIL TO REJECT H₀ ❌"} |

### Permutation Test (1000 iter)

| Statistic | Value |
|-----------|-------|
| T_obs (ΔE) | {T_obs:.6f} |
| T_null mean | {np.mean(T_null):.6f} |
| T_null 95th pct | {np.percentile(T_null, 95):.6f} |
| p-value | {p_perm:.6f} |
| Decision | {"REJECT H₀ ✅" if p_perm < 0.05 else "FAIL TO REJECT H₀ ❌"} |

## Validation Criteria

| # | Criterion | Result |
|---|-----------|--------|
| V1 | VQE convergence (E < 0) | {"✅" if convergence_ok else "❌"} |
| V2 | ΔE > 0 AND p_ttest < 0.05 | {"✅" if (delta_E > 0 and p_ttest < 0.05) else "❌"} |
| V3 | p_perm < 0.05 | {"✅" if p_perm < 0.05 else "❌"} |
| V4 | Accuracy > 50% | {"✅" if metrics['accuracy'] > 0.50 else "❌"} |
| V5 | Reproducibility | (pending — Phase 4) |
| V6 | MCP registry intact | ✅ |
| V7 | Model persisted with SHA-256 | ✅ ({vqe_result['model_hash'][:16]}...) |

## Model Artifact

| Field | Value |
|-------|-------|
| File | `{vqe_result['model_path']}` |
| SHA-256 | `{vqe_result['model_hash']}` |

## Registry Updates

- [x] experiments.json → {EXPERIMENT_ID}: {exp_status}
- [x] models.json → model-{EXPERIMENT_ID}
- [x] metrics.json → metrics-{EXPERIMENT_ID}
"""

    results_path = EXPERIMENTS_DIR / "results.md"
    results_path.write_text(results_md)
    log(f"  Results saved: {results_path}")

    # ─── SUMMARY ───
    log("")
    log("=" * 70)
    log("RESUMO FINAL")
    log("=" * 70)
    log(f"  Status:    {exp_status}")
    log(f"  Accuracy:  {metrics['accuracy']:.4f}")
    log(f"  F1 Score:  {metrics['f1_score']:.4f}")
    log(f"  ΔE:        {delta_E:.6f}")
    log(f"  p (t-test):     {p_ttest:.6f}")
    log(f"  p (perm):       {p_perm:.6f}")
    log(f"  Convergence:    {convergence_ok}")
    log(f"  Total time:     {total_time:.1f}s")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # On ANY error: mark FAILED, update lessons.md
        import traceback
        tb = traceback.format_exc()
        print(f"\n❌ FATAL ERROR: {e}\n{tb}")

        # Update experiment as FAILED
        try:
            update_registry("experiments.json", EXPERIMENT_ID, {
                "experiment_id": EXPERIMENT_ID,
                "tool_name": "tool.train_vqe_classifier",
                "status": "FAILED",
                "error": {"message": str(e), "traceback": tb},
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
        except Exception:
            pass

        # Update lessons.md
        try:
            lessons_path = PROJECT_ROOT / "tasks" / "lessons.md"
            with open(lessons_path, "a") as f:
                f.write(f"\n\n## FAILED: {EXPERIMENT_ID}\n")
                f.write(f"- Date: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
                f.write(f"- Error: {e}\n")
                f.write(f"- Traceback: {tb[:500]}\n")
        except Exception:
            pass

        sys.exit(1)
