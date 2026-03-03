# VQE Phase 1 — Results

**experiment_id**: VQE_PHASE1_BINARY
**Status**: COMPLETED_WITH_LIMITATIONS
**Executed**: 2026-03-02T21:41:15-0300

---

## Execution Summary

| Field | Value |
|-------|-------|
| Duration | 2.9s |
| Training time | 1.4s |
| Inference time | 1.3s |
| Status | COMPLETED_WITH_LIMITATIONS |

## Dataset

| Field | Value |
|-------|-------|
| Source | PlantVillage (local) |
| Classes | Tomato___Bacterial_spot, Tomato___healthy |
| Samples (train) | 400 |
| Samples (test) | 100 |
| PCA components | 8 |
| PCA variance explained | 0.5286 |
| Dataset hash | 70db8c290626a336 |

## VQE Training Results

### Per-Class Energies

| Class | Label | Optimal Energy E*_c |
|-------|-------|---------------------|
| 0 | Tomato___Bacterial_spot | -43.052406 |
| 1 | Tomato___healthy | -41.768523 |

### Energy Difference

```
ΔE = |E*_0 - E*_1| = 1.388341
```

## Classification Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.6400** |
| **F1 Score** | **0.6400** |
| Precision | 0.6400 |
| Recall | 0.6400 |

## Confusion Matrix

```
               Predicted
            | class_0  | class_1  |
Actual ─────┼──────────┼──────────┤
  class_0   |     32   |     18   |
  class_1   |     18   |     32   |
```

## Energy Statistics

| Metric | Class 0 | Class 1 |
|--------|---------|---------|
| Mean E (test) | -42.285265 | -41.492792 |
| Std E (test) | 13.264927 | 12.251265 |
| Var E (test) | 175.958293 | 150.093498 |
| n samples | 50 | 50 |

## Statistical Tests

### Welch's t-test

| Statistic | Value |
|-----------|-------|
| t-statistic | -0.3072 |
| p-value | 0.759337 |
| Decision | FAIL TO REJECT H₀ ❌ |

### Permutation Test (1000 iter)

| Statistic | Value |
|-----------|-------|
| T_obs (ΔE) | 0.792474 |
| T_null mean | 2.078171 |
| T_null 95th pct | 5.248608 |
| p-value | 0.746254 |
| Decision | FAIL TO REJECT H₀ ❌ |

## Validation Criteria

| # | Criterion | Result |
|---|-----------|--------|
| V1 | VQE convergence (E < 0) | ✅ |
| V2 | ΔE > 0 AND p_ttest < 0.05 | ❌ |
| V3 | p_perm < 0.05 | ❌ |
| V4 | Accuracy > 50% | ✅ |
| V5 | Reproducibility | (pending — Phase 4) |
| V6 | MCP registry intact | ✅ |
| V7 | Model persisted with SHA-256 | ✅ (d24127db9fb563ab...) |

## Model Artifact

| Field | Value |
|-------|-------|
| File | `/mnt/c/Users/USER/Quantum_AgriClassifier_QAC/models/vqe_classifier_VQE_PHASE1_BINARY.pkl` |
| SHA-256 | `d24127db9fb563ab7fd03370ede8fb4b958939b33d2db6fc998a4e204df70b8b` |

## Registry Updates

- [x] experiments.json → VQE_PHASE1_BINARY: COMPLETED_WITH_LIMITATIONS
- [x] models.json → model-VQE_PHASE1_BINARY
- [x] metrics.json → metrics-VQE_PHASE1_BINARY
