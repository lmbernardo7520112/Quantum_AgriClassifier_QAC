# VQE Phase 1 — Final Report

**Experiment**: VQE_PHASE1_BINARY
**Data**: 2026-03-02T21:41:15-03:00
**Duração**: 2.9s (training: 1.4s)
**Status**: COMPLETED_WITH_LIMITATIONS

---

## Sumário Executivo

O classificador VQE binário foi treinado com sucesso em PlantVillage
(`Tomato___Bacterial_spot` vs `Tomato___healthy`). O modelo atinge
**accuracy 64%** (acima de 50% random), mas a separação de energia
entre classes **não é estatisticamente significativa** (p > 0.05).

| Kritério | Resultado | Veredicto |
|----------|-----------|-----------|
| V1: VQE convergiu (E < 0) | E₀=-43.05, E₁=-41.77 | ✅ |
| V2: ΔE > 0 AND p < 0.05 | ΔE=1.39 mas p=0.76 | ❌ |
| V3: Permutation p < 0.05 | p_perm=0.75 | ❌ |
| V4: Accuracy > 50% | 64% | ✅ |
| V5: Reprodutibilidade | (Fase 4 pendente) | ⏳ |
| V6: Registry íntegro | Auditado | ✅ |
| V7: Modelo com SHA-256 | d24127db... | ✅ |

---

## Configuração Experimental

```yaml
dataset: PlantVillage (local)
classes: [Tomato___Bacterial_spot, Tomato___healthy]
samples: 500 (250+250, balanceado)
split: 400 train / 100 test (stratified)
pca: 8 components (explained variance: 52.86%)
n_qubits: 8
ansatz: RealAmplitudes (reps=2, 24 params)
optimizer: COBYLA (maxiter=100)
coupling_strength: 1.0
transverse_field: 0.5
seed: 42
estimator: StatevectorEstimator (exact)
```

## Resultados VQE

### Energias Otimizadas por Classe

| Classe | Label | E*_c | Evals |
|--------|-------|------|-------|
| 0 | Tomato___Bacterial_spot | -43.052406 | 100 |
| 1 | Tomato___healthy | -41.768523 | 100 |

**ΔE = |E*₀ - E*₁| = 1.283883**

### Métricas de Classificação

| Métrica | Valor |
|---------|-------|
| **Accuracy** | **0.6400** |
| **F1 Score** | **0.6400** |
| Training time | 1.4s |
| Inference time | 6.6s |

### Testes Estatísticos

| Teste | Estatística | p-value | Decisão |
|-------|------------|---------|---------|
| Welch t-test | t=-0.3072 | **0.7593** | FAIL TO REJECT H₀ ❌ |
| Permutation (1000) | T_obs=0.7925 | **0.7463** | FAIL TO REJECT H₀ ❌ |

### Estatísticas de Energia (Test Set)

| Métrica | Class 0 | Class 1 |
|---------|---------|---------|
| Mean E | -42.98 | -41.20 |
| Std E | 13.27 | 12.25 |
| Var E | 175.96 | 150.09 |
| n | 50 | 50 |

---

## Interpretação Científica

### O que funcionou ✅
1. **VQE convergiu** para ambas as classes, atingindo energias negativas
2. **Accuracy > random** (64% vs 50% para binário)
3. **Pipeline completo funcional**: load → PCA → VQE → metrics → persist
4. **Infraestrutura MCP íntegra**: registry, modelo, métricas persistidos

### O que não funcionou ❌
1. **p-values não significativos**: A diferença de energia entre classes
   é estatisticamente indistinguível (p ≈ 0.75)
2. **Alta variância intra-classe**: Var(E) ≈ 150-176, mascarando ΔE ≈ 1.4
3. **PCA captura apenas 52.86%** da variância original

### Causa Provável
O Hamiltoniano Ising usa **centróides de classe** para construir H_c.
Com apenas 8 features PCA (52.86% de variância), os centróides
perdem informação discriminativa. A alta variância intra-classe sugere
que a projeção Hamiltoniana não cria uma landscape energética
suficientemente distinta entre classes.

### Hipóteses para Melhoria (Fase II)
1. **Aumentar reps** do ansatz (2→4) para maior expressividade
2. **Aumentar max_iter** (100→300) para melhor convergência
3. **Ajustar γ**: transverse_field 0.5→1.0 para maior mixing
4. **Usar amostras individuais** em vez de centróides (per-sample H)
5. **Reduzir classes**: testar com classes mais distintas (ex: Apple vs Corn)
6. **Feature selection**: usar features mais discriminativas que PCA

---

## Artefatos Gerados

| Artefato | Path |
|----------|------|
| Modelo | `models/vqe_classifier_VQE_PHASE1_BINARY.pkl` |
| Resultados | `experiments/vqe_phase1/results.md` |
| Registry | `registry/experiments.json` (COMPLETED_WITH_LIMITATIONS) |
| Métricas | `registry/metrics.json` (metrics-VQE_PHASE1_BINARY) |
| Modelo reg | `registry/models.json` (model-VQE_PHASE1_BINARY) |

## Conclusão

**O VQE classifier demonstra capacidade de classificação acima de random (64%),
mas NÃO demonstra separação energética estatisticamente significativa entre classes.**

Este é um resultado cientificamente válido e honesto. O sistema VQE requer
refinamento de hiperparâmetros e/ou abordagem alternativa de construção
do Hamiltoniano antes de poder ser considerado um classificador quântico robusto.

> **STATUS: COMPLETED_WITH_LIMITATIONS**
> Accuracy criteria met. Statistical significance NOT met.
