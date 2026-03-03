# VQE Phase 1 — Results Template

**Status**: NOT EXECUTED — Template
**experiment_id**: VQE_PHASE1_BINARY

---

## Execution Summary

| Field | Value |
|-------|-------|
| Start time | (TBD) |
| End time | (TBD) |
| Duration | (TBD) |
| Status | PLANNED |

## Dataset

| Field | Value |
|-------|-------|
| Source | PlantVillage (local) |
| Classes | Tomato___healthy, Tomato___Bacterial_spot |
| Samples (train) | (TBD) |
| Samples (test) | (TBD) |
| PCA components | 8 |
| Dataset hash (SHA-256) | (TBD) |

## VQE Training Results

### Per-Class Energies

| Class | Centroid Energy E*_c | Iterations | Converged? |
|-------|---------------------|------------|------------|
| class_0 (healthy) | (TBD) | (TBD) | (TBD) |
| class_1 (bacterial) | (TBD) | (TBD) | (TBD) |

### Energy Difference

```
ΔE = |E*_0 - E*_1| = (TBD)
```

## Classification Metrics

| Metric | Value |
|--------|-------|
| Accuracy | (TBD) |
| F1 Score (weighted) | (TBD) |
| Precision | (TBD) |
| Recall | (TBD) |
| Training time (s) | (TBD) |
| Inference time (s) | (TBD) |

## Confusion Matrix

```
              Predicted
            | healthy | bacterial |
Actual ─────┼─────────┼───────────┤
  healthy   |  (TBD)  |   (TBD)   |
  bacterial |  (TBD)  |   (TBD)   |
```

## Intra-Class Energy Variance

| Class | Mean E | Std E | Var E |
|-------|--------|-------|-------|
| healthy | (TBD) | (TBD) | (TBD) |
| bacterial | (TBD) | (TBD) | (TBD) |

## Model Artifact

| Field | Value |
|-------|-------|
| File | `models/vqe_classifier_(experiment_id).pkl` |
| SHA-256 | (TBD) |
| Size | (TBD) |

## Registry Updates

- [ ] experiments.json updated
- [ ] models.json updated
- [ ] metrics.json updated
- [ ] context.json updated
