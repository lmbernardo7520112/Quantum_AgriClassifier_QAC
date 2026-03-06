# VQE Phase II — Final Report

**Status**: RESULTADO INCONCLUSIVO
**Generated**: 2026-03-03T10:37:11-0300
**Total Experiments**: 36
**Completed**: 15
**Failed**: 21

---

## 1. Experimental Design

### 1.1 Hypothesis

> Exists a parametric VQE configuration that demonstrates statistically significant
> superiority over classical baseline for PlantVillage binary classification
> (Tomato___Bacterial_spot vs Tomato___healthy)?

### 1.2 Parametric Grid

Three parameters were systematically varied:

- **Optimizer**: COBYLA, SPSA, L-BFGS-B
- **Ansatz Depth**: reps ∈ {2, 4, 6}
- **Hamiltonian Parameters**: γ ∈ {0.1, 0.5, 1.0}, h ∈ {0.1, 0.5, 1.0}

Total: 12 configurations × 3 seeds = 36 experiments.

### 1.3 Acceptance Criteria

| # | Criterion | Threshold |
|---|-----------|-----------|
| C1 | Accuracy Gap | > 3% over classical baseline |
| C2 | p-value | < 0.05 (t-test or permutation) |
| C3 | ΔE Significance | Cohen's d > 0.5 |
| C4 | NISQ Robustness | Degradation < 30% |
| C5 | Reproducibility | σ(accuracy) < 1% |

---

## 2. Phase I Reference

| Metric | Phase I Value |
|--------|--------------|
| Accuracy | 0.64 |
| F1 Score | 0.64 |
| ΔE | 1.3883409611783861 |
| p-value (t-test) | 0.7593371778855762 |
| p-value (permutation) | 0.7462537462537463 |
| Status | COMPLETED_WITH_LIMITATIONS |

---

## 3. Consolidated Results

| # | Optimizer | Reps | γ | h | Acc (μ±σ) | F1 (μ) | ΔE (μ) | Cohen's d | p-ttest | p-perm | Time (s) |
|---|-----------|------|---|---|-----------|--------|--------|----------|---------|--------|----------|
| 1 | cobyla | 2 | 0.5 | 0.5 | 0.6867±0.0287 | 0.6765 | 0.6559 | 0.2111 | 0.0344 | 0.0280 | 1.3 |
| 2 | L-BFGS-B | 6 | 0.5 | 0.5 | 0.6300±0.0748 | 0.6277 | 1.1448 | 0.1752 | 0.2577 | 0.2567 | 77.5 |
| 3 | L-BFGS-B | 6 | 1.0 | 1.0 | 0.6033±0.0624 | 0.6022 | 1.5951 | 0.1409 | 0.2954 | 0.3047 | 75.0 |
| 4 | L-BFGS-B | 4 | 0.5 | 0.5 | 0.5933±0.0903 | 0.5906 | 1.2567 | 0.2281 | 0.1265 | 0.1329 | 57.4 |
| 5 | spsa | 2 | 0.5 | 0.5 | 0.5267±0.0694 | 0.4754 | 3.4766 | 1.4255 | 0.0000 | 0.0010 | 1.4 |
| 6 | cobyla | 4 | 0.1 | 0.5 | FAILED | — | — | — | — | — | — |
| 7 | cobyla | 4 | 0.5 | 0.1 | FAILED | — | — | — | — | — | — |
| 8 | cobyla | 4 | 0.5 | 0.5 | FAILED | — | — | — | — | — | — |
| 9 | cobyla | 4 | 0.5 | 1.0 | FAILED | — | — | — | — | — | — |
| 10 | cobyla | 4 | 1.0 | 0.5 | FAILED | — | — | — | — | — | — |
| 11 | cobyla | 6 | 0.5 | 0.5 | FAILED | — | — | — | — | — | — |
| 12 | spsa | 4 | 0.5 | 0.5 | FAILED | — | — | — | — | — | — |


---

## 4. Best Configuration

| Parameter | Value |
|-----------|-------|
| Config ID | `cobyla_reps2_g0.5_h0.5` |
| Optimizer | cobyla |
| Ansatz Reps | 2 |
| γ (coupling) | 0.5 |
| h (transverse) | 0.5 |
| Accuracy (μ±σ) | 0.6867 ± 0.0287 |
| F1 Score (μ) | 0.6765 |
| ΔE (μ) | 0.6559 |
| Cohen's d (μ) | 0.2111 |
| p-value (best) | 0.027972 |
| Training time (μ) | 1.3s |

---

## 5. Superiority Criteria Evaluation

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| C1 | Accuracy Gap > 3% | ✅ PASS | Δ = 0.0467 (0.6867 - 0.6400) |
| C2 | p-value < 0.05 | ✅ PASS | p_ttest_min = 0.034435, p_perm_min = 0.027972 |
| C3 | ΔE Statistically Significant | ❌ FAIL | Cohen's d = 0.2111 |
| C4 | NISQ Robustness < 30% | ⚠️ N/A | Not evaluated (requires NISQ simulation) |
| C5 | Reproducibility < 1% | ❌ FAIL | σ(accuracy) = 0.028674 |

**Criteria Passed**: 2/4

---

## 6. Sensitivity Analysis

### 6.1 Optimizer Sensitivity

Comparison across optimizers (holding other parameters at default):

- **cobyla**: mean acc = 0.6867, range = [0.6867, 0.6867]
- **L-BFGS-B**: mean acc = 0.6089, range = [0.5933, 0.6300]
- **spsa**: mean acc = 0.5267, range = [0.5267, 0.5267]

### 6.2 Depth Sensitivity

Comparison across ansatz depths:

- **reps=2**: mean acc = 0.6067, range = [0.5267, 0.6867]
- **reps=4**: mean acc = 0.5933, range = [0.5933, 0.5933]
- **reps=6**: mean acc = 0.6167, range = [0.6033, 0.6300]

---

## 7. Scientific Conclusion

### Verdict

**RESULTADO INCONCLUSIVO**

### Justification

Some criteria were satisfied but not all. Further investigation with expanded parameter ranges, alternative Hamiltonian formulations, or larger datasets may be warranted.

### Baseline Comparison

- Classical baseline accuracy: 0.64
- Best quantum accuracy: 0.6866666666666665
- Best quantum config: `cobyla_reps2_g0.5_h0.5`

---

## 8. Registry Integrity

All experiments were registered atomically in the MCP registry:

- Each experiment has a unique `experiment_id`
- All models are persisted with SHA-256 hashes
- All metrics are recorded in `experiments/vqe_phase2/`
- No Phase I resources were modified

---

> *Generated automatically by `quantum/vqe_analysis.py`*
> *Quantum AgriClassifier — VQE Phase II*
