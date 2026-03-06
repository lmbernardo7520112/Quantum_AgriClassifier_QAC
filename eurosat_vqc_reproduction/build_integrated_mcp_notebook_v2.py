import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4, "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [l + "\\n" for l in text.split('\\n')]})

def add_code(text):
    notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\\n" for l in text.split('\\n')]})

add_md("""# VQE vs VQC Variational Classifier - EuroSat Edition (MCP Managed)
Esta versão é estruturada como um **Cliente Orquestrador MCP**, delegando o treinamento pesado ao servidor oficial `Model Context Protocol` do QAC. Esta arquitetura respeita os princípios de **Clean Code** e **Specification Driven Development**, garantindo a imutabilidade, isolamento de falhas e padronização das ferramentas na porta nativa 8000.""")

add_code("""import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time

# Configuração do MCP Server Raiz (Porta Oficial do QAC)
MCP_URL = "http://127.0.0.1:8000"

def mcp_call(endpoint: str, payload: dict):
    t0 = time.time()
    resp = requests.post(f"{MCP_URL}/{endpoint}", json=payload)
    resp.raise_for_status()
    print(f"✅ [MCP {endpoint}] ({time.time() - t0:.2f}s) OK")
    return resp.json()

try:
    health = requests.get(f"{MCP_URL}/health").json()
    print("Conexão com servidor MCP pronta:\\n", json.dumps(health, indent=2))
except:
    print("⚠️ Servidor MCP Raiz não detectado. Execute python -m mcp_server.server na raiz do projeto primeiro.")
""")

add_md("## Ingestão de Dados (EuroSat) via MCP")
add_code("""print("--- Inicializando Projeto MCP ---")
init_res = mcp_call("tools/call", {"tool_name": "tool.initialize_project", "arguments": {}})

print("\\n--- Carregando EuroSAT (AnnualCrop vs Forest) ---")
payload_load = {
    "tool_name": "tool.load_dataset",
    "arguments": {
        "dataset_name": "eurosat_rgb",
        "seed": 42, 
        "max_samples": 500
    }
}
dataset_res = mcp_call("tools/call", payload_load)
dataset_id = dataset_res.get("resource", {}).get("resource_id")
print(f"► Dataset Hash Criptográfico: {dataset_res.get('dataset_hash')}")
print(f"► Resource ID: {dataset_id}")
""")

add_md("## Treinamento de Modelos via Orquestração")
add_code("""print("\\n--- Baseline Clássico (Logistic Regression) ---")
payload_baseline = {
    "tool_name": "tool.run_baseline_logreg",
    "arguments": {"dataset_resource_id": dataset_id, "seed": 42}
}
base_res = mcp_call("tools/call", payload_baseline)
print(f"Métricas Logistic Regression: {json.dumps(base_res.get('metrics', {}), indent=2)}")

print("\\n--- Algoritmo Quântico 1: VQE (Variational Quantum Eigensolver) ---")
payload_vqe = {
    "tool_name": "tool.train_vqe_manual",
    "arguments": {
        "dataset_resource_id": dataset_id,
        "n_qubits": 4,
        "seed": 42,
        "max_iter": 50
    }
}
vqe_res = mcp_call("tools/call", payload_vqe)
print(f"Métricas VQE: {json.dumps(vqe_res.get('metrics', {}), indent=2)}")

print("\\n--- Algoritmo Quântico 2: VQC (Variational Quantum Classifier) ---")
payload_vqc = {
    "tool_name": "tool.train_vqc_manual",
    "arguments": {
        "dataset_resource_id": dataset_id,
        "n_qubits": 4,
        "seed": 42,
        "max_iter": 50
    }
}
vqc_res = mcp_call("tools/call", payload_vqc)
print(f"Métricas VQC: {json.dumps(vqc_res.get('metrics', {}), indent=2)}")
""")

add_md("## Avaliação Consolidada (Duelo Estrutural)")
add_code("""metrics_base = base_res.get('metrics', {})
metrics_vqe = vqe_res.get('metrics', {})
metrics_vqc = vqc_res.get('metrics', {})

comparison = pd.DataFrame({
    'Modelo': ['Baseline (Logistic Regression)', 'VQE (Ising)', 'VQC (Cross-Entropy)'],
    'Accuracy': [metrics_base.get('accuracy'), metrics_vqe.get('accuracy'), metrics_vqc.get('accuracy')],
    'F1 Score': [metrics_base.get('f1_score'), metrics_vqe.get('f1_score'), metrics_vqc.get('f1_score')],
    'Tempo Treino (s)': [metrics_base.get('train_time'), metrics_vqe.get('train_time'), metrics_vqc.get('train_time')]
})

print("\\n=======================================================")
print("        COMPARAÇÃO CONSOLIDADA (EUROSAT MCP Raiz)      ")
print("=======================================================")
print(comparison.to_string(index=False, float_format='%.4f'))

plt.figure(figsize=(10, 5))
methods = comparison['Modelo'].tolist()
acc = comparison['Accuracy'].tolist()
bars = plt.bar(methods, acc, color=['#888888', '#3b82f6', '#f59e0b'], edgecolor='black', width=0.4)
plt.ylabel('Acurácia')
plt.title('Comparação de Desempenho - Workflow MCP Integrado Nativo')
plt.ylim(0, 1)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center')

plt.tight_layout()
plt.show()
""")

with open("/home/leonardomaximinobernardo/My_projects/Quantum_AgriClassifier_QAC/eurosat_vqc_reproduction/VQE_vs_VQC_EuroSat_Integrated.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
print("Notebook nativo atualizado para a versão gerenciada pelo MCP principal.")
