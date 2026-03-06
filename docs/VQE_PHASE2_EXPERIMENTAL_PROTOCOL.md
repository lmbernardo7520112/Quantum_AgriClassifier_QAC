# VQE Phase II — Experimental Protocol

**Version**: 1.0.0
**Created**: 2026-03-03
**Status**: READY FOR EXECUTION

---

## 1. Hypothesis

> **H₀**: No parametric VQE configuration achieves statistically significant superiority over the classical SVM baseline for PlantVillage binary classification.

> **H₁**: There exists at least one parametric VQE configuration that demonstrates superiority satisfying all 5 formal criteria simultaneously.

## 2. Dataset

| Property | Value |
|----------|-------|
| Dataset | PlantVillage (HuggingFace) |
| Classes | `Tomato___Bacterial_spot`, `Tomato___healthy` |
| Max samples | 500 per class |
| Split | 60% train / 20% val / 20% test (stratified) |
| PCA | 8 components (= n_qubits) |
| Base seed | 42 |

## 3. Parametric Grid

### 3.1 Parameter Space

| Parameter | Symbol | Values | Justification |
|-----------|--------|--------|---------------|
| Optimizer | `opt` | COBYLA, SPSA, L-BFGS-B | Gradient-free vs approximate gradient vs quasi-Newton |
| Ansatz depth | `reps` | 2, 4, 6 | Expressivity vs trainability tradeoff |
| Coupling strength | `γ` | 0.1, 0.5, 1.0 | ZZ interaction strength in Ising Hamiltonian |
| Transverse field | `h` | 0.1, 0.5, 1.0 | X-field strength controlling tunneling |

### 3.2 Configurations (12)

| # | Optimizer | Reps | γ | h | Rationale |
|---|-----------|------|---|---|-----------|
| 1 | COBYLA | 2 | 0.5 | 0.5 | Phase I replication at lower depth |
| 2 | COBYLA | 4 | 0.5 | 0.5 | Moderate depth baseline |
| 3 | COBYLA | 6 | 0.5 | 0.5 | High depth exploration |
| 4 | COBYLA | 4 | 0.1 | 0.5 | Weak coupling limit |
| 5 | COBYLA | 4 | 1.0 | 0.5 | Strong coupling limit |
| 6 | COBYLA | 4 | 0.5 | 0.1 | Weak transverse field |
| 7 | COBYLA | 4 | 0.5 | 1.0 | Strong transverse field |
| 8 | SPSA | 2 | 0.5 | 0.5 | Stochastic optimizer @ low depth |
| 9 | SPSA | 4 | 0.5 | 0.5 | Stochastic optimizer @ moderate depth |
| 10 | L-BFGS-B | 4 | 0.5 | 0.5 | Quasi-Newton @ moderate depth |
| 11 | L-BFGS-B | 6 | 0.5 | 0.5 | Quasi-Newton @ high depth |
| 12 | L-BFGS-B | 6 | 1.0 | 1.0 | Full expressivity + strong Hamiltonian |

### 3.3 Seeds

Each configuration runs with 3 independent seeds: `{42, 123, 7}`

**Total experiments**: 12 × 3 = **36**

## 4. Metrics per Experiment

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| F1 Score | Weighted F1 score |
| ΔE | Mean energy separation between classes |
| Intra-class variance | Variance of energy evaluations within each class |
| Cohen's d | Standardized effect size |
| p-value (t-test) | Welch's t-test on class energy distributions |
| p-value (permutation) | Non-parametric test (1000 permutations) |
| Training time | Wall-clock time for VQE optimization |
| Convergence | Whether final energies are negative |

## 5. Superiority Criteria

All 5 criteria must be simultaneously satisfied:

| # | Criterion | Threshold | Rationale |
|---|-----------|-----------|-----------|
| C1 | Accuracy gap | Acc_Q > Acc_Classical + 3% | Practical significance |
| C2 | p-value | < 0.05 | Statistical significance |
| C3 | ΔE significance | Cohen's d > 0.5 | Medium effect size |
| C4 | NISQ robustness | Degradation < 30% | Hardware viability |
| C5 | Reproducibility | σ(accuracy) < 1% | Scientific reliability |

## 6. Governance Requirements

1. Each experiment creates a unique `experiment_id`
2. Full context recorded: seed, dataset_hash, backend
3. Models persisted with SHA-256 verification
4. Metrics registered atomically
5. No Phase I resources modified or overwritten
6. Experiments ledger updated atomically

## 7. Execution Command

```bash
# Validate (dry run):
PYTHONPATH=. python scripts/run_vqe_phase2.py --dry-run

# Execute:
PYTHONPATH=. python scripts/run_vqe_phase2.py
```

## 8. Expected Outputs

```
experiments/vqe_phase2/
├── VQE_PHASE2_<config>_s<seed>.json    (36 files)
├── consolidated_results.json
docs/
├── VQE_PHASE2_FINAL_REPORT.md          (auto-generated)
registry/
├── experiments.json                     (updated with 36 entries)
├── models.json                          (updated with model entries)
├── metrics.json                         (updated with metric entries)
```
