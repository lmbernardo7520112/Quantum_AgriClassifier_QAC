import json
import os

path = "/home/leonardomaximinobernardo/Downloads/quantum_camp_tech/VQE_vs_VQC_Variational_Classifier_Leonardo_Maximino_Bernardo_EuroSat.ipynb"

# Ler o notebook local (JSON nativo)
with open(path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Reposicionar a lógica de leitura na célula correta (Célula 2)
# Vamos alterar o texto orientando a pasta downloads e lidar com a falta de dados gracefully
# O script `build_notebook.py` anterior usava try/except, mas falhava silenciosamente porque `y_full` não estava definido na exception
# Vamos arrumar a lógica de extração na célula de código e no markdown

new_markdown_cell = """## 2. Carregamento e Pré-processamento de Dados (EuroSat)
Nesta versão independente, o dataset não provém ativamente de um servidor. Portanto, é imperativo que você garanta que os embeddings extraídos do projeto original (`embeddings.npy` e `labels.npy`) do **EuroSat** estejam baixados em sua máquina.

**Instruções Estruturais (SDD/Clean Pathing)**: 
A prática padrão é ter os dados repousando num diretório seguro como `~/Downloads/quantum_camp_tech/DataSetEuroSat/` ou `~/My_projects/Quantum_AgriClassifier_QAC/datasets/eurosat`. Se o notebook não encontrar os dados no caminho configurado abaixo (`DATA_PATH` e `LABELS_PATH`), a execução abortará para não gerar regressões com variáveis não instanciadas."""

new_code_cell = """import sys

# === ATENÇÃO MUDANÇA DE CAMINHOS ===
# Ajuste aqui para apontar para o diretório de embeddings local da sua máquina.
# Por padrão, assume-se a estrutura interna gerada durante o projeto QAC no diretório root datasets:
BASE_DIR = "/home/leonardomaximinobernardo/My_projects/Quantum_AgriClassifier_QAC/datasets/eurosat"
DATA_PATH = f"{BASE_DIR}/embeddings.npy"
LABELS_PATH = f"{BASE_DIR}/labels.npy"

if not os.path.exists(DATA_PATH) or not os.path.exists(LABELS_PATH):
    print(f"❌ ERRO CRÍTICO: Arquivos de embeddings não encontrados no caminho:\\n-> {BASE_DIR}")
    print("\\nPor favor, garanta que extraiu o dataset EuroSat antes de rodar os modelos quânticos.\\n(Ex: copie a pasta EuroSat pré-processada para a raiz do QAC local).")
    # Para a execução de forma elegante em vez de dar NameError lá na frente
    sys.exit("DataSet EuroSat ausente. Abortando kernel de segurança.")

X_full = np.load(DATA_PATH)
y_full = np.load(LABELS_PATH)
print(f"✅ Dados originais EuroSat carregados com sucesso!")
print(f"Shape de X: {X_full.shape}, Shape de y: {y_full.shape}")
    
# Filtrar para Classificação Binária (se o dataset for multiclasse)
# EuroSat - Classes: 0: AnnualCrop vs 1: Forest (exemplo padrão NISQ)
binary_mask = (y_full == 0) | (y_full == 1)
X_bin = X_full[binary_mask]
y_bin = y_full[binary_mask]

print(f"Dados Binários (Ano/Floresta): Shape de X: {X_bin.shape}, Shape de y: {y_bin.shape}")

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

# Identificar as células corretas e substituí-las
notebook["cells"][2]["source"] = [l + "\\n" for l in new_markdown_cell.split('\\n')]
notebook["cells"][3]["source"] = [l + "\\n" for l in new_code_cell.split('\\n')]

with open(path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook Independente (Standalone) Consertado com try/catch e instruções baseadas em SDD.")
