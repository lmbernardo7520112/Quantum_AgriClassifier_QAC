import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\\n" for line in text.split('\\n')]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\\n" for line in text.split('\\n')]
    })

text0 = """# VQE vs VQC Variational Classifier - EuroSat Edition
Esta é uma reprodução rigorosa do notebook de referência `VQE_vs_VQC_Variational_Classifier_Leonardo_Maximino_Bernardo_improved_v3.ipynb` mas aplicada ao dataset **EuroSat**.
Este notebook é a versão **Integrada** rodando dentro da estrutura do `Quantum_AgriClassifier_QAC`, aproveitando as vantagens da arquitetura orientada a serviços (Model Context Protocol - MCP) do projeto.

## 1. Imports e Configuração Básica"""
add_markdown(text0)

code1 = """import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from scipy.optimize import minimize as scipy_minimize

# Imports do Qiskit (atualizados para 2.x conforme referência)
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, zz_feature_map, real_amplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

# Configurações globais
SEED = 42
np.random.seed(SEED)
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")
N_QUBITS = 4 # Conformidade com EuroSat default
"""
add_code(code1)

text2 = """## 2. Carregamento e Pré-processamento de Dados (EuroSat)
De acordo com a metodologia de referência, vamos carregar as embeddings já extraídas do EuroSat, aplicar PCA para reduzir a dimensionalidade para um tamanho viável (`N_QUBITS=4`) e escalar usando `minmax_scale` no intervalo [-1, 1]."""
add_markdown(text2)

code2 = """# Carregar Embeddings do EuroSat
DATA_PATH = "/home/leonardomaximinobernardo/My_projects/Quantum_AgriClassifier_QAC/data/eurosat/embeddings.npy"
LABELS_PATH = "/home/leonardomaximinobernardo/My_projects/Quantum_AgriClassifier_QAC/data/eurosat/labels.npy"

try:
    X_full = np.load(DATA_PATH)
    y_full = np.load(LABELS_PATH)
    print(f"Dados originais carregados com sucesso!")
    print(f"Shape de X: {X_full.shape}, Shape de y: {y_full.shape}")
except Exception as e:
    print(f"Erro ao carregar os dados. Certifique-se de que os embeddings existem em {DATA_PATH}.")
    print(e)
    
# Filtrar para Classificação Binária (se o dataset for multiclasse)
# EuroSat - Classes: 0: AnnualCrop vs 1: Forest (exemplo padrão QAC)
binary_mask = (y_full == 0) | (y_full == 1)
X_bin = X_full[binary_mask]
y_bin = y_full[binary_mask]

print(f"Dados Binários: Shape de X: {X_bin.shape}, Shape de y: {y_bin.shape}")

# Divisão Treino/Teste
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_bin, y_bin, test_size=0.3, random_state=SEED, stratify=y_bin
)

print(f"Tamanho do Conjunto de Treino: {len(X_train_raw)}")
print(f"Tamanho do Conjunto de Teste: {len(X_test_raw)}")

# Aplicação do PCA para N_QUBITS
pca = PCA(n_components=N_QUBITS, random_state=SEED)
X_train_pca = pca.fit_transform(X_train_raw)
X_test_pca = pca.transform(X_test_raw)

# Escalonamento [-1, 1]
X_train_scaled = minmax_scale(X_train_pca, feature_range=(-1, 1))
X_test_scaled = minmax_scale(X_test_pca, feature_range=(-1, 1))

print(f"Formato final dos dados quânticos (Train): {X_train_scaled.shape}")
print(f"Variância explicada pelo PCA: {sum(pca.explained_variance_ratio_):.4f}")
"""
add_code(code2)

text3 = """## 3. Baseline Clássico — Logistic Regression
No experimento em conformidade com o notebook original, usamos **Logistic Regression** (não SVM)."""
add_markdown(text3)

code3 = """baseline_logreg = LogisticRegression(max_iter=1000, random_state=SEED)
start_time = time.time()
baseline_logreg.fit(X_train_scaled, y_train)
train_time_logreg = time.time() - start_time

y_pred_test_logreg = baseline_logreg.predict(X_test_scaled)
acc_base_eurosat = accuracy_score(y_test, y_pred_test_logreg)
f1_base_eurosat = f1_score(y_test, y_pred_test_logreg)

print(f"--- Resultados Baseline (Logistic Regression) ---")
print(f"Tempo de treinamento: {train_time_logreg:.2f} s")
print(f"Accuracy: {acc_base_eurosat:.4f}  |  F1: {f1_base_eurosat:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred_test_logreg))
"""
add_code(code3)

text4 = """## 4. Algoritmo Quântico 1: VQE (Variational Quantum Eigensolver)
Adaptado rigorosamente a partir da referência para atuar como classificador baseado em energia."""
add_markdown(text4)

code4 = """def build_hamiltonian(X_class, gamma=0.5, h=0.5):
    '''Constrói Hamiltoniano de Ising (Centróide + Mixing).'''
    centroid = np.mean(X_class, axis=0)
    n_qubits = len(centroid)
    paulis, coeffs = [], []
    for i in range(n_qubits):
        z_term = ["I"] * n_qubits
        z_term[i] = "Z"
        paulis.append("".join(z_term))
        coeffs.append(gamma * centroid[i])
    for i in range(n_qubits):
        x_term = ["I"] * n_qubits
        x_term[i] = "X"
        paulis.append("".join(x_term))
        coeffs.append(h)
    return SparsePauliOp(paulis, coeffs)

def evaluate_energy(params, ansatz, hamiltonian, estimator):
    pub = (ansatz, [hamiltonian], [params])
    job = estimator.run([pub])
    return float(job.result()[0].data.evs[0])

def run_vqe(H, reps=2, optimizer_name="COBYLA", max_iter=50, seed=42):
    ansatz = RealAmplitudes(num_qubits=H.num_qubits, reps=reps)
    estimator = StatevectorEstimator(seed=seed)
    num_params = ansatz.num_parameters
    np.random.seed(seed)
    x0 = np.random.uniform(0, 2 * np.pi, num_params)
    
    eval_count = [0]
    def cost_fn(params):
        eval_count[0] += 1
        return evaluate_energy(params, ansatz, H, estimator)
        
    method = optimizer_name if optimizer_name != "SPSA" else "Nelder-Mead"
    result = scipy_minimize(cost_fn, x0, method=method, options={"maxiter": max_iter})
    return result.fun, result.x, eval_count[0]

# --- Treinamento do VQE (EuroSat) ---
X0_es = X_train_scaled[y_train == 0]
X1_es = X_train_scaled[y_train == 1]
H0_es = build_hamiltonian(X0_es)
H1_es = build_hamiltonian(X1_es)

print("Treinando VQE paras as classes...")
t0 = time.time()
E0_opt, theta0, evals0 = run_vqe(H0_es)
E1_opt, theta1, evals1 = run_vqe(H1_es)
vqe_train_time = time.time() - t0

print(f"Classe 0: E0* = {E0_opt:.6f} ({evals0} iters)")
print(f"Classe 1: E1* = {E1_opt:.6f} ({evals1} iters)")

# --- Inferência do VQE ---
print("\\nInferindo VQE...")
t_inf = time.time()
est_inf = StatevectorEstimator(seed=SEED)
ansatz_inf = RealAmplitudes(num_qubits=N_QUBITS, reps=2)
y_pred_vqe = []

for x in X_test_scaled:
    H_x = build_hamiltonian(x.reshape(1, -1))
    e0 = evaluate_energy(theta0, ansatz_inf, H_x, est_inf)
    e1 = evaluate_energy(theta1, ansatz_inf, H_x, est_inf)
    y_pred_vqe.append(0 if e0 < e1 else 1)

vqe_inf_time = time.time() - t_inf
acc_vqe_es = accuracy_score(y_test, y_pred_vqe)
f1_vqe_es = f1_score(y_test, y_pred_vqe)

print(f"--- Resultados VQE (EuroSat) ---")
print(f"Accuracy: {acc_vqe_es:.4f}  |  F1: {f1_vqe_es:.4f}")
print(f"Tempo treino: {vqe_train_time:.1f}s | Inferência: {vqe_inf_time:.1f}s")
"""
add_code(code4)

text5 = """## 5. Algoritmo Quântico 2: VQC (Variational Quantum Classifier)
Implementação manual do forward pass e cross-entropy baseada nas primitivas do Qiskit 2.x, para contornar limitações do pacote ML, exatamente como no notebook de referência."""
add_markdown(text5)

code5 = """# Componentes do circuito VQC usando funções compatíveis qiskit 2.x
feature_map_vqc = zz_feature_map(feature_dimension=N_QUBITS, reps=2)
ansatz_vqc = real_amplitudes(num_qubits=N_QUBITS, reps=2)

fm_params = list(feature_map_vqc.parameters)
ansatz_params = list(ansatz_vqc.parameters)

vqc_circuit = feature_map_vqc.compose(ansatz_vqc)
vqc_circuit.measure_all()
sampler_vqc = StatevectorSampler(seed=SEED)

def vqc_predict_batch(params, X_batch):
    pubs = []
    for p_val in X_batch:
        parameter_bindings = {fm_param: x_val for fm_param, x_val in zip(fm_params, p_val)}
        parameter_bindings.update({ansatz_param: theta_val for ansatz_param, theta_val in zip(ansatz_params, params)})
        pubs.append((vqc_circuit, parameter_bindings))

    job = sampler_vqc.run(pubs)
    results = job.result()
    probs = []
    for res in results:
        counts = res.data.meas.get_counts()
        total_shots = sum(counts.values())
        prob_cls1 = 0.0
        for bitstring, count in counts.items():
            if bitstring.count('1') % 2 != 0:
                prob_cls1 += count
        probs.append(prob_cls1 / total_shots)
    return np.array(probs)

def vqc_cross_entropy_loss(params, X_tr, y_tr):
    probs_1 = vqc_predict_batch(params, X_tr)
    probs_1 = np.clip(probs_1, 1e-10, 1 - 1e-10)
    return -np.mean(y_tr * np.log(probs_1) + (1 - y_tr) * np.log(1 - probs_1))

# --- Treino VQC (EuroSat) ---
np.random.seed(SEED)
initial_vqc_params = np.random.uniform(0, 2*np.pi, ansatz_vqc.num_parameters)

print("Treinando VQC (COBYLA, maxiter=50)...")
t0 = time.time()
vqc_result = scipy_minimize(
    vqc_cross_entropy_loss, initial_vqc_params, args=(X_train_scaled, y_train),
    method='COBYLA', options={'maxiter': 50}
)
vqc_train_time = time.time() - t0
vqc_opt_params = vqc_result.x

# --- Inferência VQC ---
t_inf = time.time()
probs_pred = vqc_predict_batch(vqc_opt_params, X_test_scaled)
y_pred_vqc = (probs_pred > 0.5).astype(int)
vqc_inf_time = time.time() - t_inf

acc_vqc_es = accuracy_score(y_test, y_pred_vqc)
f1_vqc_es = f1_score(y_test, y_pred_vqc)

print(f"--- Resultados VQC (EuroSat) ---")
print(f"Accuracy: {acc_vqc_es:.4f}  |  F1: {f1_vqc_es:.4f}")
print(f"Tempo treino: {vqc_train_time:.1f}s | Inferência: {vqc_inf_time:.1f}s")
"""
add_code(code5)

text6 = """## 6. Comparativo Final (EuroSat)"""
add_markdown(text6)

code6 = """comparison = pd.DataFrame({
    'Dataset': ['EuroSat']*3,
    'Método': ['Baseline (LR)', 'VQE', 'VQC'],
    'Accuracy': [acc_base_eurosat, acc_vqe_es, acc_vqc_es],
    'F1': [f1_base_eurosat, f1_vqe_es, f1_vqc_es]
})

print("=======================================================")
print("        COMPARAÇÃO CONSOLIDADA (EUROSAT)               ")
print("=======================================================")
print(comparison.to_string(index=False, float_format='%.4f'))

fig, ax = plt.subplots(figsize=(8, 5))
methods = ['Baseline (LR)', 'VQE', 'VQC']
acc_vals = [acc_base_eurosat, acc_vqe_es, acc_vqc_es]
colors = ['#888888', '#3b82f6', '#f59e0b']

ax.bar(methods, acc_vals, color=colors, edgecolor='black', width=0.5)
ax.set_ylabel('Accuracy')
ax.set_title('Comparação de Performance (Dataset: EuroSat)')
ax.set_ylim(0, 1)

for i, val in enumerate(acc_vals):
    ax.text(i, val + 0.02, f"{val:.4f}", ha='center')

plt.tight_layout()
plt.show()
"""
add_code(code6)

output_path = "/home/leonardomaximinobernardo/My_projects/Quantum_AgriClassifier_QAC/eurosat_vqc_reproduction/VQE_vs_VQC_EuroSat_Integrated.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
print("Notebook final criado com sucesso via dict JSON nativo!")
