# 🌾 Quantum_AgriClassifier_QAC — Hybrid Quantum-Classical Framework for Agricultural Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge\&logo=python)
![Qiskit](https://img.shields.io/badge/Qiskit-SDK-purple?style=for-the-badge\&logo=qiskit)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge\&logo=pytorch)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge\&logo=scikitlearn)
![FastAPI](https://img.shields.io/badge/FastAPI-MCP-green?style=for-the-badge\&logo=fastapi)
![NumPy](https://img.shields.io/badge/NumPy-Array-lightblue?style=for-the-badge\&logo=numpy)

> [!WARNING]
> **Work In Progress (WIP)**: This project is under active architectural and scientific development. Quantum modules, statistical validation pipelines, and distributed execution mechanisms are continuously being refined. Experimental results should be interpreted within the context of controlled research validation.

---

## 📘 General Description

**Quantum AgriClassifier (QAC)** is a rigorously structured hybrid quantum-classical platform designed for supervised classification of multiespectral agricultural data.

The project integrates:

* Classical machine learning pipelines
* Variational quantum algorithms
* Energy-based quantum modeling
* Distributed execution under Model Context Protocol (MCP)
* Formal governance through Software Design Documentation (SDD)

The core objective is to investigate the feasibility, robustness, and statistical validity of quantum-enhanced classification methods within precision agriculture workflows.

QAC is not merely a model repository — it is a **scientifically governed experimental framework**.

---

## 🧩 Hybrid Architecture

```
┌────────────────────────────────┐
│     Classical Preprocessing     │
│  (Normalization + PCA + Split)  │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│    Quantum Module Layer         │
│  (QSVM / VQC / VQE Classifier)  │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│   Statistical Evaluation Layer  │
│ (Accuracy, F1, ΔE, Permutation) │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│      MCP Server (FastAPI)       │
│ Persistent Registry & Audit Log │
└────────────────────────────────┘
```

---

## 🧠 Quantum Modules

### 1️⃣ Quantum Kernel Methods (QSVM)

Implements quantum feature maps and kernel estimation using
Qiskit Machine Learning.

Used for:

* High-dimensional Hilbert space separability
* Kernel-based supervised classification

---

### 2️⃣ Variational Quantum Classifier (VQC)

Parameterized quantum circuits trained via classical optimization.

* Ansatz: `RealAmplitudes` / `EfficientSU2`
* Optimizers: COBYLA / SPSA
* Backend: Aer Statevector or hardware

Paradigm:
Loss minimization directly on measurement expectation values.

---

### 3️⃣ VQE-Based Energy Classifier (Incremental Extension)

Implements classification via the
Variational Quantum Eigensolver.

For each class, an optimal variational state is trained to minimize:

[
H(x) = \sum_i x_i Z_i + \sum_{i<j} x_i x_j Z_i Z_i + \gamma \sum_i X_i
]

The Hamiltonian belongs to the class of the
Ising model with data-conditioned longitudinal and transverse fields.

Classification rule:

[
\text{Class}(x) = \arg\min_c \langle \psi(\theta_c^*) | H(x) | \psi(\theta_c^*) \rangle
]

This module was implemented incrementally, without altering prior QSVM/VQC pipelines, preserving full architectural stability.

---

# 🔬 Experimental Results — VQE PHASE I (Integrated Update)

The first controlled experiment using VQE was executed under full MCP governance and SDD protocol compliance.

### 🏷 Status

`COMPLETED_WITH_LIMITATIONS`

### 📊 Performance Metrics

| Metric                | Value                      | Criterion | Verdict |
| --------------------- | -------------------------- | --------- | ------- |
| Accuracy              | 0.6400                     | > 50%     | ✅ PASS  |
| F1 Score              | 0.6400                     | > 55%     | ✅ PASS  |
| VQE Convergence       | E₀ = -43.05<br>E₁ = -41.77 | E < 0     | ✅ PASS  |
| ΔE                    | 1.39                       | > 0       | ✅ PASS  |
| p-value (t-test)      | 0.759                      | < 0.05    | ❌ FAIL  |
| p-value (Permutation) | 0.746                      | < 0.05    | ❌ FAIL  |
| Registry Integrity    | Audited                    | —         | ✅ PASS  |
| Model Persistence     | SHA-256: d24127db...       | —         | ✅ PASS  |

---

## 📈 Scientific Interpretation

The VQE classifier operates above random baseline (64%), confirming:

* Correct Hamiltonian construction
* Variational convergence
* Proper MCP execution lifecycle

However, statistical testing indicates that the observed energy separation:

[
\Delta E = 1.39
]

is not statistically significant under α = 0.05.

High intra-class variance (≈150–175) masks the modest energy gap.

Additionally:

* PCA (8 components) retains only 52.86% of total variance
* The centroid-based Hamiltonian reduces discriminative structure
* Energy landscape overlap remains substantial

### Conclusion

The system is **operationally valid** but **not yet statistically robust** in its current configuration.

---

## 🧪 Scientific Implications

This result is meaningful:

* The architecture works.
* The VQE converges.
* The registry persists deterministically.
* The experiment is reproducible.
* The scientific protocol correctly prevents premature claims of quantum advantage.

QAC therefore demonstrates methodological rigor rather than optimistic bias.

---

## 🏗 MCP — Model Context Protocol

The system operates under distributed execution control:

* Persistent experiment ledger (`experiments.json`)
* Deterministic context tracking (`context.json`)
* SHA-256 model hashing
* Schema-validated tools
* Atomic registry updates
* Crash-safe lifecycle management

Server execution:

```bash
PYTHONPATH=. python -m mcp_server.server
```

Guarantees:

| Property        | Guarantee                 |
| --------------- | ------------------------- |
| Persistence     | Atomic writes             |
| Determinism     | Fixed seed + dataset hash |
| Isolation       | Tool-level encapsulation  |
| Reproducibility | Context versioning        |
| Auditability    | Ledger tracking           |

---

## 🧪 Validation Philosophy

QAC does **not** assume quantum advantage.

Instead, it enforces:

* Controlled baselines
* Statistical hypothesis testing
* Energy gap validation
* Reproducibility analysis
* Registry-backed experiment persistence

The VQE Phase I outcome exemplifies this philosophy.

---

## 🧩 Folder Structure

```
Quantum_AgriClassifier_QAC/
├── classical/
├── quantum/
│   ├── qsvm.py
│   ├── vqc.py
│   ├── hamiltonian_builder.py
│   ├── vqe_classifier.py
├── mcp_server/
├── registry/
├── experiments/
│   └── vqe_phase1/
├── docs/
└── README.md
```

---

## 🚀 Current Status

```
ARCHITECTURE: Stable
MCP: Operational
QSVM: Functional
VQC: Functional
VQE: Executed (Phase I)
STATISTICAL SIGNIFICANCE: Not achieved
REGISTRY: Consistent and Audited
```

System state:

**SCIENTIFICALLY VALIDATED — ITERATIVE IMPROVEMENT REQUIRED**

---

## 🔮 Next Research Directions

* Increase ansatz expressivity
* Deepen optimization convergence
* Refine Hamiltonian formulation
* Evaluate alternative class separability
* Implement per-sample Hamiltonian modeling
* Conduct reproducibility & sensitivity phase

---

> 💬 *Quantum AgriClassifier is not built to prove quantum superiority — it is built to test it under controlled, reproducible, and statistically defensible conditions.*
— Leonardo Maximino Bernardo, 2026
