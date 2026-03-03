# VQE Incremental Specification — QAC Extension

## 1. Objective

Add **VQE-based quantum classification** to QAC as an alternative quantum pipeline.
Unlike VQC (which uses a quantum classifier directly), the VQE approach uses the
Variational Quantum Eigensolver to find optimal Hamiltonian parameters that encode
class separation in eigenvalue spectra.

## 2. Architecture

```
                 ┌──────────────────────────┐
                 │     PCA-reduced data      │
                 │   (n_qubits features)     │
                 └────────┬─────────────────┘
                          │
            ┌─────────────▼──────────────┐
            │   hamiltonian_builder.py    │
            │                            │
            │  data → Ising Hamiltonian  │
            │  class-conditioned H(x)    │
            │  SparsePauliOp output      │
            └─────────────┬──────────────┘
                          │
            ┌─────────────▼──────────────┐
            │    vqe_classifier.py       │
            │                            │
            │  VQE minimizes ⟨ψ|H|ψ⟩    │
            │  Optimal θ* per class      │
            │  Classification by E(θ*)   │
            └─────────────┬──────────────┘
                          │
            ┌─────────────▼──────────────┐
            │  tool.train_vqe_classifier │
            │                            │
            │  MCP tool wrapper          │
            │  Auto-PCA, metrics, hash   │
            └────────────────────────────┘
```

## 3. New Files

| File | Purpose | Dependencies |
|------|---------|-------------|
| `quantum/hamiltonian_builder.py` | Build data-conditioned Ising Hamiltonians | `qiskit.quantum_info` |
| `quantum/vqe_classifier.py` | VQE-based classifier with per-class energy minimization | `hamiltonian_builder`, `qiskit_algorithms` |
| `docs/VQE_INCREMENTAL_SPEC.md` | This specification | — |
| `experiments/vqe_phase1.md` | Experiment template (not executed) | — |
| `docs/VQE_EXTENSION_READY.md` | Readiness confirmation report | — |

## 4. Hamiltonian Construction

The Hamiltonian `H(x)` is a parameterized Ising model:

```
H(x) = Σᵢ xᵢ Zᵢ + Σᵢⱼ xᵢxⱼ ZᵢZⱼ + Σᵢ Xᵢ
```

Where:
- `xᵢ` = PCA-reduced feature i (angle-encoded)
- `Zᵢ` = Pauli-Z on qubit i
- `ZᵢZⱼ` = Ising coupling between qubits i,j
- `Xᵢ` = transverse field term for mixing

This creates a data-dependent energy landscape where different classes
produce different ground state energies.

## 5. Classification Approach

### Training:
1. For each class `c`, collect training samples `{x | y=c}`
2. Build class-conditioned Hamiltonians `H_c(x̄)` using class centroids
3. Run VQE to find optimal ansatz parameters `θ*_c` minimizing `⟨ψ(θ)|H_c|ψ(θ)⟩`
4. Store `{θ*_c, E*_c}` per class

### Inference:
1. For new sample `x`, build `H(x)`
2. Evaluate `E_c = ⟨ψ(θ*_c)|H(x)|ψ(θ*_c)⟩` for each class
3. Classify as `argmin_c E_c`

## 6. Integration Points

- **Input**: `DatasetResult` with PCA-reduced features (same as QSVM/VQC)
- **Output**: model artifact + metrics (same schema as existing quantum tools)
- **Registry**: `resource.model` + `resource.metrics` (standard)
- **Context**: seed + backend + dataset_hash (standard via ContextManager)

## 7. Constraints

- Maximum 10 qubits (NISQ compatible)
- Aer simulator primary, IBM Quantum optional
- COBYLA optimizer (default), SPSA available
- Feature dimension must equal n_qubits (PCA enforced)

## 8. SDD Compliance

| Requirement | How Met |
|-------------|---------|
| Schema before code | `tool.train_vqe_classifier` schema defined |
| Metrics defined | accuracy, f1, training_time, vqe_energy |
| Preconditions | `dataset_loaded` |
| Postconditions | `resource.model_registered`, `resource.metrics_registered` |
| Determinism | seed + dataset_hash in context |
