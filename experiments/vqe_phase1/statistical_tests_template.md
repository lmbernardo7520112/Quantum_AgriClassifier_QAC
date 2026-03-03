# VQE Phase 1 — Statistical Tests Template

**Status**: NOT EXECUTED — Template
**experiment_id**: VQE_PHASE1_BINARY

---

## Test 1: Welch's t-test (Two-Sample, Two-Sided)

### Setup

```
H₀: μ_E₀ = μ_E₁   (mean energies equal across classes)
H₁: μ_E₀ ≠ μ_E₁   (mean energies differ)

α = 0.05
Test: scipy.stats.ttest_ind(E_class0, E_class1, equal_var=False)
```

### Variables

- **E_class0**: Array of energies ⟨ψ(θ*₀)|H(xᵢ)|ψ(θ*₀)⟩ for all xᵢ where yᵢ = 0
- **E_class1**: Array of energies ⟨ψ(θ*₁)|H(xᵢ)|ψ(θ*₁)⟩ for all xᵢ where yᵢ = 1

### Results

| Statistic | Value |
|-----------|-------|
| t-statistic | (TBD) |
| p-value | (TBD) |
| df (Welch-Satterthwaite) | (TBD) |
| Mean E₀ | (TBD) |
| Mean E₁ | (TBD) |
| Std E₀ | (TBD) |
| Std E₁ | (TBD) |
| n₀ | (TBD) |
| n₁ | (TBD) |

### Decision

```
p < 0.05? → (TBD)
Decision: (TBD) — Reject H₀ / Fail to reject H₀
```

---

## Test 2: Permutation Test (≥1000 iterations)

### Setup

```
H₀: Label assignment is independent of VQE energy
H₁: Label assignment is related to VQE energy

n_permutations = 1000
Test statistic: T = |mean(E_class0) - mean(E_class1)|
```

### Algorithm

```python
T_obs = |mean(E_class0) - mean(E_class1)|

T_null = []
for k in range(1000):
    y_perm = np.random.permutation(y_test)
    E_perm_0 = E_all[y_perm == 0]
    E_perm_1 = E_all[y_perm == 1]
    T_null.append(|mean(E_perm_0) - mean(E_perm_1)|)

p_perm = (sum(T_null >= T_obs) + 1) / (1000 + 1)
```

### Results

| Statistic | Value |
|-----------|-------|
| T_obs (ΔE observed) | (TBD) |
| T_null mean | (TBD) |
| T_null std | (TBD) |
| T_null 95th percentile | (TBD) |
| p-value (permutation) | (TBD) |
| n_permutations | 1000 |

### Decision

```
p_perm < 0.05? → (TBD)
Decision: (TBD) — Reject H₀ / Fail to reject H₀
```

---

## Combined Conclusion

| Test | p-value | Decision |
|------|---------|----------|
| Welch t-test | (TBD) | (TBD) |
| Permutation | (TBD) | (TBD) |

### Overall Statistical Verdict

```
Both tests reject H₀? → (TBD)
Conclusion: (TBD)
```

---

## Assumptions and Limitations

1. **t-test**: Assumes approximately normal distribution of energies
   - Validated by: (TBD — Shapiro-Wilk test on energy distributions)
2. **Permutation test**: Non-parametric, no distributional assumptions
3. **Both tests**: Energy evaluations use the same trained θ* parameters
4. **Multiple comparisons**: Not applicable (only 2 tests, same hypothesis)
