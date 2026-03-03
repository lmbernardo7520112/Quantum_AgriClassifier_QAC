# VQE Phase I — Plano de Execução Formal

**Documento**: VQE_PHASE1_EXECUTION_PLAN.md
**Status**: PLANEJADO — Aguardando autorização para execução
**Data**: 2026-03-02
**Governança**: SDD + MCP

---

## 1. Formulação Matemática do Hamiltoniano

### 1.1 Modelo Ising Paramétrico

O hamiltoniano data-condicionado é definido como:

```
H(x) = Σᵢ₌₁ⁿ xᵢ Zᵢ + γ Σᵢ<ⱼ xᵢxⱼ ZᵢZⱼ + h Σᵢ₌₁ⁿ Xᵢ
```

Onde:
- **x ∈ ℝⁿ** = vetor de features PCA-reduzido e normalizado para [0, 2π]
- **Zᵢ** = operador Pauli-Z no qubit i
- **ZᵢZⱼ** = acoplamento Ising entre qubits i, j
- **Xᵢ** = campo transverso (termo de mixing)
- **γ** = `coupling_strength` (hiperparâmetro, default = 1.0)
- **h** = `transverse_field` (hiperparâmetro, default = 0.5)
- **n** = `n_qubits` (dimensão PCA = número de qubits)

### 1.2 Hamiltoniano de Classe

Para cada classe c, o hamiltoniano é construído a partir do **centróide**:

```
x̄_c = (1/|C_c|) Σ_{x∈C_c} x
H_c = H(x̄_c)
```

Onde C_c é o conjunto de amostras da classe c.

### 1.3 Otimização VQE

O VQE minimiza a energia do estado variacional:

```
E*_c = min_θ ⟨ψ(θ)|H_c|ψ(θ)⟩
```

Onde |ψ(θ)⟩ é o ansatz parametrizado (RealAmplitudes com 2 repetições).

### 1.4 Regra de Classificação

Para um novo dado x:

```
ŷ = argmin_c ⟨ψ(θ*_c)|H(x)|ψ(θ*_c)⟩
```

A classe com menor energia de expectativa é atribuída.

---

## 2. Hipótese Experimental

### H₀ (Hipótese Nula)

> O classificador VQE **não** produz separação de energia significativamente
> diferente entre classes. A diferença média de energia entre classes
> ΔE = |E*_class0 − E*_class1| **não é** estatisticamente significativa.

Formalmente: `p(ΔE = 0 | H₀) > α` com α = 0.05.

### H₁ (Hipótese Alternativa)

> O classificador VQE produz ΔE > 0 estatisticamente significativo (p < 0.05),
> indicando que o mapeamento Hamiltoniano captura informação discriminativa
> nos dados agrícolas.

### Justificativa Estatística

- Classes selecionadas: **2 classes** binário para primeira validação
- Classes: `Tomato___healthy` (class 0) vs `Tomato___Bacterial_spot` (class 1)
  - **Razão**: mesmo gênero (Tomate), diferença patológica clara, balanced
- Tamanho amostral: **max_samples=500** (250 por classe) para viabilidade computacional
- Nível de significância: α = 0.05
- Poder estatístico desejado: β ≥ 0.80

---

## 3. Configuração Completa do Experimento

```yaml
# experiments/vqe_phase1/config.yaml
experiment_id: VQE_PHASE1_BINARY
dataset:
  name: plantvillage
  source: local → C:\Users\USER\Downloads\Quantum_AgriClassifier_QAC_dataset
  classes_filter: ["Tomato___healthy", "Tomato___Bacterial_spot"]
  max_samples: 500
  split_ratio: 0.8 (400 treino / 100 teste)
  seed: 42

preprocessing:
  pca_components: 8
  normalization: "[0, 2π] feature-wise"

quantum:
  n_qubits: 8
  ansatz: real_amplitudes
  ansatz_reps: 2
  optimizer: cobyla
  max_iter: 100
  coupling_strength: 1.0   # γ
  transverse_field: 0.5    # h
  backend: aer_statevector

reproducibility:
  seed: 42
  dataset_hash: SHA-256 (computado automaticamente)
  context_version: registrado no context.json
```

---

## 4. Métricas

### 4.1 Métricas Primárias

| Métrica | Definição | Critério |
|---------|-----------|----------|
| **Accuracy** | (TP+TN)/N | > 60% (baseline binário = 50%) |
| **F1 Score** | 2·P·R/(P+R) weighted | > 0.55 |
| **ΔE** | \|E\*\_class0 − E\*\_class1\| | > 0 (significativo) |
| **p-value** | t-test bilateral | < 0.05 |

### 4.2 Métricas Secundárias

| Métrica | Definição |
|---------|-----------|
| VQE Energy (class 0) | E\*₀ = min_θ ⟨ψ(θ)\|H₀\|ψ(θ)⟩ |
| VQE Energy (class 1) | E\*₁ = min_θ ⟨ψ(θ)\|H₁\|ψ(θ)⟩ |
| Variância intra-classe | Var(E\|y=c) para amostras de teste |
| Training time | Tempo total de otimização VQE |
| Convergência | Número de iterações até convergência |

---

## 5. Testes Estatísticos

### 5.1 Teste t de Student (bilateral)

```
H₀: μ_E₀ = μ_E₁  (energias médias iguais)
H₁: μ_E₀ ≠ μ_E₁  (energias médias diferentes)

t = (x̄_E₀ - x̄_E₁) / √(s²_E₀/n₀ + s²_E₁/n₁)
df = Welch-Satterthwaite
```

- Avaliado sobre energias ⟨ψ(θ\*_c)|H(x)|ψ(θ\*_c)⟩ para cada x no test set
- Cada amostra produz duas energias (uma por θ\* de cada classe)
- Se p < 0.05: rejeitar H₀

### 5.2 Teste de Permutação (≥1000 iterações)

```
Para k = 1..1000:
  1. Permutar labels y aleatoriamente
  2. Recomputar ΔE_perm = |mean(E_perm_0) - mean(E_perm_1)|
  3. Acumular distribuição nula

p_perm = (#{ΔE_perm ≥ ΔE_obs} + 1) / (1000 + 1)
```

- Não requer distribuição Gaussiana
- Robusto para amostras pequenas

---

## 6. Critérios de Validação

| # | Critério | Condição |
|---|----------|----------|
| V1 | VQE convergiu | Ambas classes atingem energia negativa |
| V2 | ΔE significativo | ΔE > 0 AND p_ttest < 0.05 |
| V3 | Permutation test | p_perm < 0.05 |
| V4 | Accuracy > random | Acc > 50% (binário) |
| V5 | Reprodutível | Re-run com seed=42 → mesmos resultados |
| V6 | Registry íntegro | Auditoria MCP sem issues |
| V7 | Modelo persistido | .pkl existe com hash SHA-256 |

Se V1-V7 todos satisfeitos: **EXPERIMENT VALIDATED**
Se qualquer Vi falhar: **EXPERIMENT COMPLETED WITH LIMITATIONS** (documentar quais)

---

## 7. Plano de Reprodutibilidade

| Etapa | Ação |
|-------|------|
| R1 | Re-executar com seed=42 → accuracy idêntica |
| R2 | Re-executar com seed=123 → accuracy ≤ ±5% |
| R3 | Verificar dataset_hash idêntico entre runs |
| R4 | Verificar model_hash idêntico para mesma seed |
| R5 | Verificar context.json registra todas as runs |

---

## 8. Análise de Risco Técnico

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|-----------|
| VQE não converge | Média | Alto | Aumentar max_iter para 200 |
| Energias iguais (ΔE ≈ 0) | Média | Alto | Ajustar γ e h |
| Overfitting no centróide | Baixa | Médio | Validação cruzada na Fase II |
| Tempo computacional > 1h | Média | Baixo | Reduzir max_samples para 200 |
| Memória insuficiente | Baixa | Médio | Usar max_samples=200 |
| PCA perde informação | Média | Médio | Testar n_components = 4, 6, 8 |

---

## 9. Critério de Parada Científica

| Condição | Ação |
|----------|------|
| p > 0.10 | STOP — VQE não discrimina estas classes |
| Accuracy < 50% | STOP — Pior que random |
| VQE diverge (E → +∞) | STOP — Hamiltoniano mal-condicionado |
| Training > 2 horas | STOP — Computacionalmente inviável |
| Crash de servidor | STOP — Registrar em lessons.md |

---

## 10. Sequência de Execução (Fase 3 — sob comando)

```
1. tool.initialize_project  → confirm datasets
2. tool.load_dataset         → plantvillage, filter 2 classes
3. Apply PCA                 → 8 components
4. tool.train_vqe_classifier → VQE optimization
5. Compute metrics           → accuracy, F1, ΔE
6. Statistical tests         → t-test + permutation
7. Persist all               → registry + model + metrics
8. Generate report           → VQE_PHASE1_FINAL_REPORT.md
9. Reproducibility run       → same seed verification
10. Sensitivity analysis     → γ, reps variation
```

**Nenhuma destas etapas será iniciada sem autorização explícita.**
