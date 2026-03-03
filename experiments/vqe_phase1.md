# VQE Phase 1 — Experiment Template

**Status**: NOT EXECUTED — Template only
**experiment_id**: (to be assigned at execution time)

---

## Objective

Evaluate VQE-based quantum classification on PlantVillage (38 classes)
using Ising Hamiltonians and variational energy minimization.

## Configuration

```yaml
dataset: plantvillage
source: local (fallback: huggingface)
classes: 38
pca_components: 8
n_qubits: 8
ansatz: real_amplitudes
optimizer: cobyla
max_iter: 100
coupling_strength: 1.0
transverse_field: 0.5
seed: 42
backend: aer_statevector
```

## Expected Workflow

1. `tool.load_dataset` → PlantVillage, 38 classes
2. Auto-PCA to 8 features
3. `tool.train_vqe_classifier` → per-class Hamiltonian + VQE
4. `tool.compare_models` → VQE vs QSVM vs VQC vs SVM

## Success Criteria

| Metric | Target |
|--------|--------|
| Accuracy | > 30% (38 classes, random = 2.6%) |
| F1 Weighted | > 0.25 |
| Training time | < 1 hour (Aer simulator) |
| VQE convergence | All classes reach negative energy |

## Resources Expected

- `resource.model`: `vqe_classifier_{experiment_id}.pkl`
- `resource.metrics`: accuracy, f1, vqe_energies
- `registry/experiments.json`: experiment logged with COMPLETED status

## Comparison Plan

Compare with existing models:
- SVM baseline (classical)
- CNN baseline (classical)
- QSVM (quantum kernel)
- VQC (variational classifier)
- Data Re-uploading (quantum)

## Notes

- VQE is compute-intensive: per-class optimization × test sample evaluation
- Consider `max_samples=500` for initial development runs
- Full 54,306 samples only for final benchmarks
