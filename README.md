# 🌾 Quantum AgriClassifier (QAC) — v2.0

**Classificação Quântica Híbrida de Imagens Agrícolas**

> Brazil Quantum Camp — Bloco 3 Entregável

---

## 🎯 Objetivo

Investigar a viabilidade do **Variational Quantum Classifier (VQC)** para classificação de imagens de satélite agrícolas (EuroSAT), comparando com baseline clássico (SVM).

## 🏗 Arquitetura

```
EuroSAT (2 classes) → PCA (8 features) → Normalize [0,2π]
    → ZZFeatureMap(8) → RealAmplitudes(8, reps=2) → COBYLA(100)
    → VQC.fit() → VQC.predict() → Accuracy, F1, Confusion Matrix
```

## 📂 Estrutura do Projeto

```
Quantum_AgriClassifier_QAC/
├── src/qac/                    # Pacote Python principal
│   ├── config.py               # Dataclasses imutáveis (SDD)
│   ├── data_loader.py          # EuroSAT loader com filtragem
│   ├── preprocessing.py        # PCA + normalização quântica
│   ├── feature_map_factory.py  # Factory: ZZ, Z, Pauli
│   ├── ansatz_factory.py       # Factory: RealAmplitudes, EfficientSU2
│   ├── optimizer_factory.py    # Factory: COBYLA, SPSA
│   ├── vqc_classifier.py      # VQC (modelo principal)
│   ├── classical_baseline.py   # SVM baseline
│   ├── evaluation.py           # Métricas + comparação
│   └── experiment.py           # Orquestrador do pipeline
├── configs/                    # Configurações JSON
│   ├── default.json            # Config padrão
│   └── fast.json               # Config rápida (CI/testes)
├── tests/                      # TDD (unit + integration)
│   ├── unit/                   # 46 testes rápidos
│   └── integration/
├── scripts/
│   ├── run_experiment.py       # CLI runner
│   └── generate_deliverables.py # Gerador de report/slides
├── notebooks/
│   └── QAC_Bloco3_Experiment.ipynb  # Notebook standalone
├── docs/                       # Entregáveis Bloco 3
├── pyproject.toml              # Projeto instalável
└── Makefile                    # Targets padronizados
```

## 🚀 Quick Start

```bash
# Instalar
make install

# Rodar testes
make test

# Executar experimento completo
make run

# Executar versão rápida (CI)
make run-fast

# Abrir notebook
make notebook
```

## 📊 Metodologia

| Princípio | Implementação |
|-----------|--------------|
| **SDD** | Configs imutáveis (`DataConfig`, `ModelConfig`, `ExperimentConfig`) |
| **Clean Code** | `src/` layout, factory pattern, typed results |
| **TDD** | 46 unit tests + integration tests separados |
| **MCP** | Server simplificado para integração com IA |

## 📋 Entregáveis Bloco 3

| Entregável | Arquivo | Status |
|-----------|---------|--------|
| Notebook | `notebooks/QAC_Bloco3_Experiment.ipynb` | ✅ Standalone |
| Relatório | `docs/Entregavel_Bloco_3_*.docx` | ✅ 3 páginas |
| Slides | `docs/Slides_Bloco_3_*.pptx` | ✅ 5 slides |

## 📖 Referências

- Benedetti et al. (2019). Parameterized quantum circuits as ML models. QST.
- Schuld & Petruccione (2021). Machine Learning with Quantum Computers. Springer.
- Helber et al. (2019). EuroSAT: Novel Dataset for Land Use Classification. IEEE JSTARS.
- Havlíček et al. (2019). Supervised learning with quantum-enhanced feature spaces. Nature.

---

**Autor**: Leonardo Maximino Bernardo | **Equipe**: Quantum Tech | **2026**
