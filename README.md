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

```text
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

Implements quantum feature maps and kernel estimation using the ecosystem of Qiskit Machine Learning.

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
H(x) = \sum_i x_i Z_i + \sum_{i<j} x_i x_j Z_i Z_j + \gamma \sum_i X_i
]

The Hamiltonian belongs to the class of the
Ising model with data-conditioned longitudinal and transverse fields.

Classification rule:

[
\text{Class}(x) = \arg\min_c \langle \psi(\theta_c^*) | H(x) | \psi(\theta_c^*) \rangle
]

This extension was implemented incrementally, preserving all existing modules and ensuring zero regression.

---

## 🏗 MCP — Model Context Protocol

The system operates under a distributed execution architecture:

* Persistent experiment ledger (`experiments.json`)
* Deterministic context tracking (`context.json`)
* Model versioning with SHA-256 hashes
* Schema-validated tools
* Atomic registry updates
* Crash-safe execution

Server execution:

```bash
PYTHONPATH=. python -m mcp_server.server
```

Key guarantees:

| Property        | Guarantee                 |
| --------------- | ------------------------- |
| Persistence     | Atomic writes             |
| Determinism     | Fixed seed + dataset hash |
| Isolation       | Tool-level encapsulation  |
| Reproducibility | Context versioning        |
| Auditability    | Ledger tracking           |

---

## 🧪 Scientific Protocol

Experiments follow formal methodology:

* Explicit null and alternative hypotheses
* Statistical tests (t-test + permutation)
* Energy gap analysis (ΔE)
* Reproducibility validation
* Sensitivity analysis
* MCP audit verification

No experiment is considered valid without full registry persistence and reproducibility checks.

---

## 🧩 Folder Structure

```bash
Quantum_AgriClassifier_QAC/
├── classical/
│   ├── baseline.py
│   ├── data_loader.py
├── quantum/
│   ├── qsvm.py
│   ├── vqc.py
│   ├── hamiltonian_builder.py
│   ├── vqe_classifier.py
├── mcp_server/
│   ├── server.py
│   ├── execution_engine.py
│   ├── schemas.py
│   ├── vqe_tool.py
├── registry/
│   ├── experiments.json
│   ├── models.json
│   ├── metrics.json
│   ├── context.json
├── experiments/
│   └── vqe_phase1/
├── docs/
│   ├── VQE_INCREMENTAL_SPEC.md
│   ├── VQE_PHASE1_EXECUTION_PLAN.md
│   └── VQE_EXTENSION_READY.md
└── README.md
```

---

## ⚙️ Design Decisions

| Topic                   | Strategy                   | Benefit                  |
| ----------------------- | -------------------------- | ------------------------ |
| Quantum Integration     | Modular independent layers | Zero regression          |
| Registry                | JSON atomic persistence    | Crash resilience         |
| Execution               | Tool-isolated lifecycle    | Safe distributed control |
| Experiment Governance   | SDD-driven planning        | Scientific rigor         |
| Incremental Development | Non-intrusive extensions   | Architectural stability  |

---

## 🧪 Validation Philosophy

QAC does **not** claim quantum advantage a priori.

It provides:

* Controlled comparison between classical and quantum models
* Structured experimental validation
* Transparent statistical evaluation
* Engineering-grade reproducibility

---

## 🕒 Development Timeline

### 🧩 Phase 1 — Classical Foundation

* Dataset ingestion pipeline
* Baseline ML models
* Metrics and validation structure

### ⚛ Phase 2 — Variational Quantum Models

* QSVM integration
* VQC training module
* Hybrid optimization framework

### 🏗 Phase 3 — MCP Architecture

* FastAPI server
* ExecutionEngine lifecycle
* Persistent experiment ledger
* Deterministic context management

### 🔬 Phase 4 — VQE Energy-Based Extension

* Data-conditioned Ising Hamiltonian
* Per-class variational training
* Incremental tool integration
* Scientific protocol formalization

---

## 🚀 Current Status

```text
ARCHITECTURE: Stable
MCP: Operational
QSVM: Functional
VQC: Functional
VQE: Structurally Implemented
EXPERIMENTS: Controlled Mode
```

System state:

**READY FOR CONTROLLED SCIENTIFIC EXECUTION**

---

> 💬 *“QAC is not simply a quantum classifier — it is a governed experimental infrastructure for investigating the real scientific boundaries of quantum-enhanced learning in agriculture.”*
> — Leonardo Maximino Bernardo, 2026
